import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import get_current_ops
from spacy.attrs import LENGTH, ORTH
from spacy.lang.en import English
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab
from .test_underscore import clean_underscore  # noqa: F401
def test_for_no_ent_sents():
    """Span.sents() should set .sents correctly, even if Span in question is trailing and doesn't form a full
    sentence.
    """
    doc = Doc(English().vocab, words=['This', 'is', 'a', 'test.', 'ENTITY'], sent_starts=[1, 0, 0, 0, 1])
    doc.set_ents([Span(doc, 4, 5, 'WORK')])
    sents = list(doc.ents[0].sents)
    assert len(sents) == 1
    assert str(sents[0]) == str(doc.ents[0].sent) == 'ENTITY'