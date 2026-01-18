import pytest
from spacy.pipeline._parser_internals.stateclass import StateClass
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab
def test_H(doc):
    state = StateClass(doc)
    assert state.H(0) == -1
    state.add_arc(1, 0, 0)
    assert state.arcs == [{'head': 1, 'child': 0, 'label': 0}]
    assert state.H(0) == 1
    state.add_arc(3, 1, 0)
    assert state.H(1) == 3