import copy
import pickle
import re
import pytest
from mock import Mock
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_dependency_matcher_long_matches(en_vocab, doc):
    pattern = [{'RIGHT_ID': 'quick', 'RIGHT_ATTRS': {'DEP': 'amod', 'OP': '+'}}]
    matcher = DependencyMatcher(en_vocab)
    with pytest.raises(ValueError):
        matcher.add('pattern', [pattern])