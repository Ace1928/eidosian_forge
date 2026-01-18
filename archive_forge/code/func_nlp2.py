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
@pytest.fixture
def nlp2(nlp, sample_vectors):
    Language.component('test_language_vector_modification_pipe', func=vector_modification_pipe)
    Language.component('test_language_userdata_pipe', func=userdata_pipe)
    Language.component('test_language_ner_pipe', func=ner_pipe)
    add_vecs_to_vocab(nlp.vocab, sample_vectors)
    nlp.add_pipe('test_language_vector_modification_pipe')
    nlp.add_pipe('test_language_ner_pipe')
    nlp.add_pipe('test_language_userdata_pipe')
    return nlp