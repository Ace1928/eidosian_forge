import numpy
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from thinc.api import NumpyOps, Ragged, get_current_ops
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.tokens import SpanGroup
from spacy.tokens._dict_proxies import SpanGroups
from spacy.training import Example
from spacy.util import fix_random_seed, make_tempdir, registry
@pytest.mark.parametrize('name', SPANCAT_COMPONENTS)
def test_explicit_labels(name):
    nlp = Language()
    spancat = nlp.add_pipe(name, config={'spans_key': SPAN_KEY})
    assert len(spancat.labels) == 0
    spancat.add_label('PERSON')
    spancat.add_label('LOC')
    nlp.initialize()
    assert spancat.labels == ('PERSON', 'LOC')