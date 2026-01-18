import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest import approx
from spacy.lang.en import English
from spacy.scorer import PRFScore, ROCAUCScore, Scorer, _roc_auc_score, _roc_curve
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
def span_getter(doc, span_key):
    return doc.spans[span_key]