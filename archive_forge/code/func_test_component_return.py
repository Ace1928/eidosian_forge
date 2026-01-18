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
def test_component_return():
    """Test that an error is raised if components return a type other than a
    doc."""
    nlp = English()

    @Language.component('test_component_good_pipe')
    def good_pipe(doc):
        return doc
    nlp.add_pipe('test_component_good_pipe')
    nlp('text')
    nlp.remove_pipe('test_component_good_pipe')

    @Language.component('test_component_bad_pipe')
    def bad_pipe(doc):
        return doc.text
    nlp.add_pipe('test_component_bad_pipe')
    with pytest.raises(ValueError, match='instead of a Doc'):
        nlp('text')