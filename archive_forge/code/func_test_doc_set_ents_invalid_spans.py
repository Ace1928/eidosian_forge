import warnings
import weakref
import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.attrs import (
from spacy.lang.en import English
from spacy.lang.xx import MultiLanguage
from spacy.language import Language
from spacy.lexeme import Lexeme
from spacy.tokens import Doc, Span, SpanGroup, Token
from spacy.vocab import Vocab
from .test_underscore import clean_underscore  # noqa: F401
def test_doc_set_ents_invalid_spans(en_tokenizer):
    doc = en_tokenizer('Some text about Colombia and the Czech Republic')
    spans = [Span(doc, 3, 4, label='GPE'), Span(doc, 6, 8, label='GPE')]
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    with pytest.raises(IndexError):
        doc.ents = spans