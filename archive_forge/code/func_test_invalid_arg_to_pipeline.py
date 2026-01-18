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
def test_invalid_arg_to_pipeline(nlp):
    str_list = ['This is a text.', 'This is another.']
    with pytest.raises(ValueError):
        nlp(str_list)
    assert len(list(nlp.pipe(str_list))) == 2
    int_list = [1, 2, 3]
    with pytest.raises(ValueError):
        list(nlp.pipe(int_list))
    with pytest.raises(ValueError):
        nlp(int_list)