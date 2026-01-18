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
@pytest.mark.parametrize('lang,target', [('en', 'en'), ('fra', 'fr'), ('fre', 'fr'), ('iw', 'he'), ('mo', 'ro'), ('mul', 'xx'), ('no', 'nb'), ('pt-BR', 'pt'), ('xx', 'xx'), ('zh-Hans', 'zh')])
def test_blank_languages(lang, target):
    """
    Test that we can get spacy.blank in various languages, including codes
    that are defined to be equivalent or that match by CLDR language matching.
    """
    nlp = spacy.blank(lang)
    assert nlp.lang == target