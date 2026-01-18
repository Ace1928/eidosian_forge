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
@registry.callbacks(f'{name}_after_init')
def make_after_init():

    def after_init(nlp):
        nonlocal ran_after_init
        ran_after_init = True
        nlp.meta['after_init'] = 'after'
        return nlp
    return after_init