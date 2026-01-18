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
def test_evaluate_no_pipe(nlp):
    """Test that docs are processed correctly within Language.pipe if the
    component doesn't expose a .pipe method."""

    @Language.component('test_evaluate_no_pipe')
    def pipe(doc):
        return doc
    text = 'hello world'
    annots = {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0}}
    nlp = Language(Vocab())
    doc = nlp(text)
    nlp.add_pipe('test_evaluate_no_pipe')
    nlp.evaluate([Example.from_dict(doc, annots)])