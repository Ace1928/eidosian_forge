import copy
import pickle
import re
import pytest
from mock import Mock
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.issue(9263)
def test_dependency_matcher_remove(en_tokenizer):
    doc = en_tokenizer('The red book')
    doc[1].head = doc[2]
    pattern1 = [{'RIGHT_ID': 'root', 'RIGHT_ATTRS': {'ORTH': 'book'}}, {'LEFT_ID': 'root', 'RIGHT_ID': 'r', 'RIGHT_ATTRS': {'ORTH': 'red'}, 'REL_OP': '>'}]
    matcher = DependencyMatcher(en_tokenizer.vocab)
    matcher.add('check', [pattern1])
    matcher.remove('check')
    pattern2 = [{'RIGHT_ID': 'root', 'RIGHT_ATTRS': {'ORTH': 'flag'}}, {'LEFT_ID': 'root', 'RIGHT_ID': 'r', 'RIGHT_ATTRS': {'ORTH': 'blue'}, 'REL_OP': '>'}]
    matcher.add('check', [pattern2])
    matches = matcher(doc)
    assert matches == []