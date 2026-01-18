import copy
import pickle
import re
import pytest
from mock import Mock
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_dependency_matcher_pickle(en_vocab, patterns, doc):
    matcher = DependencyMatcher(en_vocab)
    for i in range(1, len(patterns) + 1):
        matcher.add('pattern' + str(i), [patterns[i - 1]])
    matches = matcher(doc)
    assert matches[0][1] == [3, 1, 2]
    assert matches[1][1] == [4, 3, 5]
    assert matches[2][1] == [4, 3, 2]
    assert matches[3][1] == [4, 3]
    assert matches[4][1] == [4, 3]
    assert matches[5][1] == [4, 8]
    b = pickle.dumps(matcher)
    matcher_r = pickle.loads(b)
    assert len(matcher) == len(matcher_r)
    matches = matcher_r(doc)
    assert matches[0][1] == [3, 1, 2]
    assert matches[1][1] == [4, 3, 5]
    assert matches[2][1] == [4, 3, 2]
    assert matches[3][1] == [4, 3]
    assert matches[4][1] == [4, 3]
    assert matches[5][1] == [4, 8]