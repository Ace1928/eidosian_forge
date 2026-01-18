from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_substitute_(self):
    assert self.cur_token_ in {'substitute', 'sub', 'reversesub', 'rsub'}
    location = self.cur_token_location_
    reverse = self.cur_token_ in {'reversesub', 'rsub'}
    old_prefix, old, lookups, values, old_suffix, hasMarks = self.parse_glyph_pattern_(vertical=False)
    if any(values):
        raise FeatureLibError('Substitution statements cannot contain values', location)
    new = []
    if self.next_token_ == 'by':
        keyword = self.expect_keyword_('by')
        while self.next_token_ != ';':
            gc = self.parse_glyphclass_(accept_glyphname=True, accept_null=True)
            new.append(gc)
    elif self.next_token_ == 'from':
        keyword = self.expect_keyword_('from')
        new = [self.parse_glyphclass_(accept_glyphname=False)]
    else:
        keyword = None
    self.expect_symbol_(';')
    if len(new) == 0 and (not any(lookups)):
        raise FeatureLibError('Expected "by", "from" or explicit lookup references', self.cur_token_location_)
    if keyword == 'from':
        if reverse:
            raise FeatureLibError('Reverse chaining substitutions do not support "from"', location)
        if len(old) != 1 or len(old[0].glyphSet()) != 1:
            raise FeatureLibError('Expected a single glyph before "from"', location)
        if len(new) != 1:
            raise FeatureLibError('Expected a single glyphclass after "from"', location)
        return self.ast.AlternateSubstStatement(old_prefix, old[0], old_suffix, new[0], location=location)
    num_lookups = len([l for l in lookups if l is not None])
    is_deletion = False
    if len(new) == 1 and isinstance(new[0], ast.NullGlyph):
        new = []
        is_deletion = True
    if not reverse and len(old) == 1 and (len(new) == 1) and (num_lookups == 0):
        glyphs = list(old[0].glyphSet())
        replacements = list(new[0].glyphSet())
        if len(replacements) == 1:
            replacements = replacements * len(glyphs)
        if len(glyphs) != len(replacements):
            raise FeatureLibError('Expected a glyph class with %d elements after "by", but found a glyph class with %d elements' % (len(glyphs), len(replacements)), location)
        return self.ast.SingleSubstStatement(old, new, old_prefix, old_suffix, forceChain=hasMarks, location=location)
    if is_deletion and len(old) == 1 and (num_lookups == 0):
        return self.ast.MultipleSubstStatement(old_prefix, old[0], old_suffix, (), forceChain=hasMarks, location=location)
    if not reverse and len(old) == 1 and (len(new) > 1) and (num_lookups == 0):
        count = len(old[0].glyphSet())
        for n in new:
            if not list(n.glyphSet()):
                raise FeatureLibError('Empty class in replacement', location)
            if len(n.glyphSet()) != 1 and len(n.glyphSet()) != count:
                raise FeatureLibError(f'Expected a glyph class with 1 or {count} elements after "by", but found a glyph class with {len(n.glyphSet())} elements', location)
        return self.ast.MultipleSubstStatement(old_prefix, old[0], old_suffix, new, forceChain=hasMarks, location=location)
    if not reverse and len(old) > 1 and (len(new) == 1) and (len(new[0].glyphSet()) == 1) and (num_lookups == 0):
        return self.ast.LigatureSubstStatement(old_prefix, old, old_suffix, list(new[0].glyphSet())[0], forceChain=hasMarks, location=location)
    if reverse:
        if len(old) != 1:
            raise FeatureLibError('In reverse chaining single substitutions, only a single glyph or glyph class can be replaced', location)
        if len(new) != 1:
            raise FeatureLibError('In reverse chaining single substitutions, the replacement (after "by") must be a single glyph or glyph class', location)
        if num_lookups != 0:
            raise FeatureLibError('Reverse chaining substitutions cannot call named lookups', location)
        glyphs = sorted(list(old[0].glyphSet()))
        replacements = sorted(list(new[0].glyphSet()))
        if len(replacements) == 1:
            replacements = replacements * len(glyphs)
        if len(glyphs) != len(replacements):
            raise FeatureLibError('Expected a glyph class with %d elements after "by", but found a glyph class with %d elements' % (len(glyphs), len(replacements)), location)
        return self.ast.ReverseChainSingleSubstStatement(old_prefix, old_suffix, old, new, location=location)
    if len(old) > 1 and len(new) > 1:
        raise FeatureLibError('Direct substitution of multiple glyphs by multiple glyphs is not supported', location)
    if len(new) != 0 or is_deletion:
        raise FeatureLibError('Invalid substitution statement', location)
    rule = self.ast.ChainContextSubstStatement(old_prefix, old, old_suffix, lookups, location=location)
    return rule