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
@pytest.mark.issue(3199)
def test_issue3199():
    """Test that Span.noun_chunks works correctly if no noun chunks iterator
    is available. To make this test future-proof, we're constructing a Doc
    with a new Vocab here and a parse tree to make sure the noun chunks run.
    """
    words = ['This', 'is', 'a', 'sentence']
    doc = Doc(Vocab(), words=words, heads=[0] * len(words), deps=['dep'] * len(words))
    with pytest.raises(NotImplementedError):
        list(doc[0:3].noun_chunks)