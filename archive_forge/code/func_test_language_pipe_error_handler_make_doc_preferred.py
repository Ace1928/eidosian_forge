import itertools
import logging
import warnings
from unittest import mock
import pytest
from thinc.api import CupyOps, NumpyOps, get_current_ops
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import find_matching_language, ignore_error, raise_error, registry
from spacy.vocab import Vocab
from .util import add_vecs_to_vocab, assert_docs_equal
@pytest.mark.xfail
@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe_error_handler_make_doc_preferred(n_process):
    """Test the error handling for make_doc"""
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        nlp = English()
        nlp.max_length = 10
        texts = ['12345678901234567890', '12345'] * 10
        with pytest.raises(ValueError):
            list(nlp.pipe(texts, n_process=n_process))
        nlp.default_error_handler = ignore_error
        docs = list(nlp.pipe(texts, n_process=n_process))
        assert len(docs) == 0