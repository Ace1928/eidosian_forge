import copy
import pickle
import re
import pytest
from mock import Mock
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_dependency_matcher_pattern_validation(en_vocab):
    pattern = [{'RIGHT_ID': 'fox', 'RIGHT_ATTRS': {'ORTH': 'fox'}}, {'LEFT_ID': 'fox', 'REL_OP': '>', 'RIGHT_ID': 'q', 'RIGHT_ATTRS': {'ORTH': 'quick', 'DEP': 'amod'}}, {'LEFT_ID': 'fox', 'REL_OP': '>', 'RIGHT_ID': 'r', 'RIGHT_ATTRS': {'ORTH': 'brown'}}]
    matcher = DependencyMatcher(en_vocab)
    matcher.add('FOUNDED', [pattern])
    with pytest.raises(ValueError):
        matcher.add('FOUNDED', pattern)
    with pytest.raises(ValueError):
        matcher.add('FOUNDED', [pattern[1:]])
    with pytest.raises(ValueError):
        pattern2 = copy.deepcopy(pattern)
        del pattern2[0]['RIGHT_ID']
        matcher.add('FOUNDED', [pattern2])
    with pytest.raises(ValueError):
        pattern2 = copy.deepcopy(pattern)
        del pattern2[1]['RIGHT_ID']
        matcher.add('FOUNDED', [pattern2])
    with pytest.raises(ValueError):
        pattern2 = copy.deepcopy(pattern)
        del pattern2[1]['RIGHT_ATTRS']
        matcher.add('FOUNDED', [pattern2])
    with pytest.raises(ValueError):
        pattern2 = copy.deepcopy(pattern)
        del pattern2[1]['LEFT_ID']
        matcher.add('FOUNDED', [pattern2])
    with pytest.raises(ValueError):
        pattern2 = copy.deepcopy(pattern)
        del pattern2[1]['REL_OP']
        matcher.add('FOUNDED', [pattern2])
    with pytest.raises(ValueError):
        pattern2 = copy.deepcopy(pattern)
        pattern2[1]['REL_OP'] = '!!!'
        matcher.add('FOUNDED', [pattern2])
    with pytest.raises(ValueError):
        pattern2 = copy.deepcopy(pattern)
        pattern2[1]['RIGHT_ID'] = 'fox'
        matcher.add('FOUNDED', [pattern2])
    with pytest.warns(UserWarning):
        pattern2 = copy.deepcopy(pattern)
        pattern2[1]['FOO'] = 'BAR'
        matcher.add('FOUNDED', [pattern2])