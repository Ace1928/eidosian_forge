import copy
import pickle
import re
import pytest
from mock import Mock
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.parametrize('left,right,op,num_matches', [('fox', 'jumped', '<', 1), ('the', 'lazy', '<', 0), ('jumped', 'jumped', '<', 0), ('fox', 'jumped', '>', 0), ('fox', 'lazy', '>', 1), ('lazy', 'lazy', '>', 0), ('fox', 'jumped', '<<', 2), ('jumped', 'fox', '<<', 0), ('the', 'fox', '<<', 2), ('fox', 'jumped', '>>', 0), ('over', 'the', '>>', 1), ('fox', 'the', '>>', 2), ('fox', 'jumped', '.', 1), ('lazy', 'fox', '.', 1), ('the', 'fox', '.', 0), ('the', 'the', '.', 0), ('fox', 'jumped', ';', 0), ('lazy', 'fox', ';', 0), ('the', 'fox', ';', 0), ('the', 'the', ';', 0), ('quick', 'fox', '.*', 2), ('the', 'fox', '.*', 3), ('the', 'the', '.*', 1), ('fox', 'jumped', ';*', 1), ('quick', 'fox', ';*', 0), ('the', 'fox', ';*', 1), ('the', 'the', ';*', 1), ('quick', 'brown', '$+', 1), ('brown', 'quick', '$+', 0), ('brown', 'brown', '$+', 0), ('quick', 'brown', '$-', 0), ('brown', 'quick', '$-', 1), ('brown', 'brown', '$-', 0), ('the', 'brown', '$++', 1), ('brown', 'the', '$++', 0), ('brown', 'brown', '$++', 0), ('the', 'brown', '$--', 0), ('brown', 'the', '$--', 1), ('brown', 'brown', '$--', 0), ('over', 'jumped', '<+', 0), ('quick', 'fox', '<+', 0), ('the', 'quick', '<+', 0), ('brown', 'fox', '<+', 1), ('quick', 'fox', '<++', 1), ('quick', 'over', '<++', 0), ('over', 'jumped', '<++', 0), ('the', 'fox', '<++', 2), ('brown', 'fox', '<-', 0), ('fox', 'over', '<-', 0), ('the', 'over', '<-', 0), ('over', 'jumped', '<-', 1), ('brown', 'fox', '<--', 0), ('fox', 'jumped', '<--', 0), ('fox', 'over', '<--', 1), ('fox', 'brown', '>+', 0), ('over', 'fox', '>+', 0), ('over', 'the', '>+', 0), ('jumped', 'over', '>+', 1), ('jumped', 'over', '>++', 1), ('fox', 'lazy', '>++', 0), ('over', 'the', '>++', 0), ('jumped', 'over', '>-', 0), ('fox', 'quick', '>-', 0), ('brown', 'quick', '>-', 0), ('fox', 'brown', '>-', 1), ('brown', 'fox', '>--', 0), ('fox', 'brown', '>--', 1), ('jumped', 'fox', '>--', 1), ('fox', 'the', '>--', 2)])
def test_dependency_matcher_ops(en_vocab, doc, left, right, op, num_matches):
    right_id = right
    if left == right:
        right_id = right + '2'
    pattern = [{'RIGHT_ID': left, 'RIGHT_ATTRS': {'LOWER': left}}, {'LEFT_ID': left, 'REL_OP': op, 'RIGHT_ID': right_id, 'RIGHT_ATTRS': {'LOWER': right}}]
    matcher = DependencyMatcher(en_vocab)
    matcher.add('pattern', [pattern])
    matches = matcher(doc)
    assert len(matches) == num_matches