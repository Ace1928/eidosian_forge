import pytest
from spacy.pipeline._parser_internals.stateclass import StateClass
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab
def test_init_state(doc):
    state = StateClass(doc)
    assert state.stack == []
    assert state.queue == list(range(len(doc)))
    assert not state.is_final()
    assert state.buffer_length() == 4