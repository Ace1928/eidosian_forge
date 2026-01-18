import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
def test_matcher_with_alignments_non_greedy(en_vocab):
    cases = [(0, 'aaab', 'a* b', [[0, 1], [0, 0, 1], [0, 0, 0, 1], [1]]), (1, 'baab', 'b a* b', [[0, 1, 1, 2]]), (2, 'aaab', 'a a a b', [[0, 1, 2, 3]]), (3, 'aaab', 'a+ b', [[0, 1], [0, 0, 1], [0, 0, 0, 1]]), (4, 'aaba', 'a+ b a+', [[0, 1, 2], [0, 0, 1, 2]]), (5, 'aabaa', 'a+ b a+', [[0, 1, 2], [0, 0, 1, 2], [0, 0, 1, 2, 2], [0, 1, 2, 2]]), (6, 'aaba', 'a+ b a*', [[0, 1], [0, 0, 1], [0, 0, 1, 2], [0, 1, 2]]), (7, 'aaaa', 'a*', [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]), (8, 'baab', 'b a* b b*', [[0, 1, 1, 2]]), (9, 'aabb', 'a* b* a*', [[1], [2], [2, 2], [0, 1], [0, 0, 1], [0, 0, 1, 1], [0, 1, 1], [1, 1]]), (10, 'aaab', 'a+ a+ a b', [[0, 1, 2, 3]]), (11, 'aaab', 'a+ a+ a+ b', [[0, 1, 2, 3]]), (12, 'aaab', 'a+ a a b', [[0, 1, 2, 3]]), (13, 'aaab', 'a+ a a', [[0, 1, 2]]), (14, 'aaab', 'a+ a a?', [[0, 1], [0, 1, 2]]), (15, 'aaaa', 'a a a a a?', [[0, 1, 2, 3]]), (16, 'aaab', 'a+ a b', [[0, 1, 2], [0, 0, 1, 2]]), (17, 'aaab', 'a+ a+ b', [[0, 1, 2], [0, 0, 1, 2]]), (18, 'aaab', 'a{2,} b', [[0, 0, 1], [0, 0, 0, 1]]), (19, 'aaab', 'a{3} b', [[0, 0, 0, 1]]), (20, 'aaab', 'a{2} b', [[0, 0, 1]]), (21, 'aaab', 'a{2,3} b', [[0, 0, 1], [0, 0, 0, 1]])]
    for case_id, string, pattern_str, results in cases:
        matcher = Matcher(en_vocab)
        doc = Doc(matcher.vocab, words=list(string))
        pattern = []
        for part in pattern_str.split():
            if part.endswith('+'):
                pattern.append({'ORTH': part[0], 'OP': '+'})
            elif part.endswith('*'):
                pattern.append({'ORTH': part[0], 'OP': '*'})
            elif part.endswith('?'):
                pattern.append({'ORTH': part[0], 'OP': '?'})
            elif part.endswith('}'):
                pattern.append({'ORTH': part[0], 'OP': part[1:]})
            else:
                pattern.append({'ORTH': part})
        matcher.add('PATTERN', [pattern])
        matches = matcher(doc, with_alignments=True)
        n_matches = len(matches)
        for _, s, e, expected in matches:
            assert expected in results, (case_id, string, pattern_str, s, e, n_matches)
            assert len(expected) == e - s