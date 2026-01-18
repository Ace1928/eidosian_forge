import copy
import pickle
import re
import pytest
from mock import Mock
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_dependency_matcher(dependency_matcher, doc, patterns):
    assert len(dependency_matcher) == 5
    assert 'pattern3' in dependency_matcher
    assert dependency_matcher.get('pattern3') == (None, [patterns[2]])
    matches = dependency_matcher(doc)
    assert len(matches) == 6
    assert matches[0][1] == [3, 1, 2]
    assert matches[1][1] == [4, 3, 5]
    assert matches[2][1] == [4, 3, 2]
    assert matches[3][1] == [4, 3]
    assert matches[4][1] == [4, 3]
    assert matches[5][1] == [4, 8]
    span = doc[0:6]
    matches = dependency_matcher(span)
    assert len(matches) == 5
    assert matches[0][1] == [3, 1, 2]
    assert matches[1][1] == [4, 3, 5]
    assert matches[2][1] == [4, 3, 2]
    assert matches[3][1] == [4, 3]
    assert matches[4][1] == [4, 3]