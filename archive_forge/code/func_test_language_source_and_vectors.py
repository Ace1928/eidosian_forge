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
def test_language_source_and_vectors(nlp2):
    nlp = Language(Vocab())
    textcat = nlp.add_pipe('textcat')
    for label in ('POSITIVE', 'NEGATIVE'):
        textcat.add_label(label)
    nlp.initialize()
    long_string = 'thisisalongstring'
    assert long_string not in nlp.vocab.strings
    assert long_string not in nlp2.vocab.strings
    nlp.vocab.strings.add(long_string)
    assert nlp.vocab.vectors.to_bytes() != nlp2.vocab.vectors.to_bytes()
    vectors_bytes = nlp.vocab.vectors.to_bytes()
    with pytest.warns(UserWarning):
        nlp2.add_pipe('textcat', name='textcat2', source=nlp)
    assert long_string in nlp2.vocab.strings
    assert nlp.vocab.vectors.to_bytes() == vectors_bytes