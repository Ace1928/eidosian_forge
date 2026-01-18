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
@pytest.mark.issue(4903)
def test_issue4903():
    """Ensure that this runs correctly and doesn't hang or crash on Windows /
    macOS."""
    nlp = English()
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('my_pipe', after='sentencizer')
    text = ['I like bananas.', 'Do you like them?', 'No, I prefer wasabi.']
    if isinstance(get_current_ops(), NumpyOps):
        docs = list(nlp.pipe(text, n_process=2))
        assert docs[0].text == 'I like bananas.'
        assert docs[1].text == 'Do you like them?'
        assert docs[2].text == 'No, I prefer wasabi.'