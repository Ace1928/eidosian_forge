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
@pytest.mark.issue(1537)
def test_issue1537():
    """Test that Span.as_doc() doesn't segfault."""
    string = 'The sky is blue . The man is pink . The dog is purple .'
    doc = Doc(Vocab(), words=string.split())
    doc[0].sent_start = True
    for word in doc[1:]:
        if word.nbor(-1).text == '.':
            word.sent_start = True
        else:
            word.sent_start = False
    sents = list(doc.sents)
    sent0 = sents[0].as_doc()
    sent1 = sents[1].as_doc()
    assert isinstance(sent0, Doc)
    assert isinstance(sent1, Doc)