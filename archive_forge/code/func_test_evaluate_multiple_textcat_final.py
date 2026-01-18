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
def test_evaluate_multiple_textcat_final(en_vocab):
    """Test that evaluate evaluates the final textcat component in a pipeline
    with more than one textcat or textcat_multilabel."""
    nlp = Language(en_vocab)
    textcat = nlp.add_pipe('textcat')
    for label in ('POSITIVE', 'NEGATIVE'):
        textcat.add_label(label)
    textcat_multilabel = nlp.add_pipe('textcat_multilabel')
    for label in ('FEATURE', 'REQUEST', 'BUG', 'QUESTION'):
        textcat_multilabel.add_label(label)
    nlp.initialize()
    annots = {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0, 'FEATURE': 1.0, 'QUESTION': 1.0, 'POSITIVE': 1.0, 'NEGATIVE': 0.0}}
    doc = nlp.make_doc('hello world')
    example = Example.from_dict(doc, annots)
    scores = nlp.evaluate([example])
    labels = nlp.get_pipe(nlp.pipe_names[-1]).labels
    for label in labels:
        assert scores['cats_f_per_type'].get(label) is not None
    for key in example.reference.cats.keys():
        if key not in labels:
            assert scores['cats_f_per_type'].get(key) is None