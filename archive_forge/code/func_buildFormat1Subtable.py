from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def buildFormat1Subtable(self, ruleset, chaining=True):
    st = self.newSubtable_(chaining=chaining)
    st.Format = 1
    st.populateDefaults()
    coverage = set()
    rulesetsByFirstGlyph = {}
    ruleAttr = self.ruleAttr_(format=1, chaining=chaining)
    for rule in ruleset.rules:
        ruleAsSubtable = self.newRule_(format=1, chaining=chaining)
        if chaining:
            ruleAsSubtable.BacktrackGlyphCount = len(rule.prefix)
            ruleAsSubtable.LookAheadGlyphCount = len(rule.suffix)
            ruleAsSubtable.Backtrack = [list(x)[0] for x in reversed(rule.prefix)]
            ruleAsSubtable.LookAhead = [list(x)[0] for x in rule.suffix]
            ruleAsSubtable.InputGlyphCount = len(rule.glyphs)
        else:
            ruleAsSubtable.GlyphCount = len(rule.glyphs)
        ruleAsSubtable.Input = [list(x)[0] for x in rule.glyphs[1:]]
        self.buildLookupList(rule, ruleAsSubtable)
        firstGlyph = list(rule.glyphs[0])[0]
        if firstGlyph not in rulesetsByFirstGlyph:
            coverage.add(firstGlyph)
            rulesetsByFirstGlyph[firstGlyph] = []
        rulesetsByFirstGlyph[firstGlyph].append(ruleAsSubtable)
    st.Coverage = buildCoverage(coverage, self.glyphMap)
    ruleSets = []
    for g in st.Coverage.glyphs:
        ruleSet = self.newRuleSet_(format=1, chaining=chaining)
        setattr(ruleSet, ruleAttr, rulesetsByFirstGlyph[g])
        setattr(ruleSet, f'{ruleAttr}Count', len(rulesetsByFirstGlyph[g]))
        ruleSets.append(ruleSet)
    setattr(st, self.ruleSetAttr_(format=1, chaining=chaining), ruleSets)
    setattr(st, self.ruleSetAttr_(format=1, chaining=chaining) + 'Count', len(ruleSets))
    return st