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
def test_language_update(nlp):
    text = 'hello world'
    annots = {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0}}
    wrongkeyannots = {'LABEL': True}
    doc = Doc(nlp.vocab, words=text.split(' '))
    example = Example.from_dict(doc, annots)
    nlp.update([example])
    with pytest.raises(TypeError):
        nlp.update(example)
    with pytest.raises(TypeError):
        nlp.update((text, annots))
    with pytest.raises(TypeError):
        nlp.update((doc, annots))
    with pytest.raises(ValueError):
        example = Example.from_dict(doc, None)
    with pytest.raises(KeyError):
        example = Example.from_dict(doc, wrongkeyannots)