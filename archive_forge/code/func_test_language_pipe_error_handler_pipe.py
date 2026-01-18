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
@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe_error_handler_pipe(en_vocab, n_process):
    """Test the error handling of a component's pipe method"""
    Language.component('my_perhaps_sentences', func=perhaps_set_sentences)
    Language.component('assert_sents_error', func=assert_sents_error)
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        texts = [f'{str(i)} is enough. Done' for i in range(100)]
        nlp = English()
        nlp.add_pipe('my_perhaps_sentences')
        nlp.add_pipe('assert_sents_error')
        nlp.initialize()
        with pytest.raises(ValueError):
            docs = list(nlp.pipe(texts, n_process=n_process, batch_size=10))
        nlp.set_error_handler(ignore_error)
        docs = list(nlp.pipe(texts, n_process=n_process, batch_size=10))
        assert len(docs) == 89