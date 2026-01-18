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
@pytest.mark.parametrize('max_positive,nr_results', [(None, 4), (1, 2), (2, 3), (3, 4), (4, 4)])
def test_make_spangroup_multilabel(max_positive, nr_results):
    fix_random_seed(0)
    nlp = Language()
    spancat = nlp.add_pipe('spancat', config={'spans_key': SPAN_KEY, 'threshold': 0.5, 'max_positive': max_positive})
    doc = nlp.make_doc('Greater London')
    ngram_suggester = registry.misc.get('spacy.ngram_suggester.v1')(sizes=[1, 2])
    indices = ngram_suggester([doc])[0].dataXd
    assert_array_equal(OPS.to_numpy(indices), numpy.asarray([[0, 1], [1, 2], [0, 2]]))
    labels = ['Thing', 'City', 'Person', 'GreatCity']
    for label in labels:
        spancat.add_label(label)
    scores = numpy.asarray([[0.2, 0.4, 0.3, 0.1], [0.1, 0.6, 0.2, 0.4], [0.8, 0.7, 0.3, 0.9]], dtype='f')
    spangroup = spancat._make_span_group_multilabel(doc, indices, scores)
    assert len(spangroup) == nr_results
    assert spangroup[0].text == 'London'
    assert spangroup[0].label_ == 'City'
    assert_almost_equal(0.6, spangroup.attrs['scores'][0], 5)
    assert spangroup[1].text == 'Greater London'
    if max_positive == 1:
        assert spangroup[1].label_ == 'GreatCity'
        assert_almost_equal(0.9, spangroup.attrs['scores'][1], 5)
    else:
        assert spangroup[1].label_ == 'Thing'
        assert_almost_equal(0.8, spangroup.attrs['scores'][1], 5)
    if nr_results > 2:
        assert spangroup[2].text == 'Greater London'
        if max_positive == 2:
            assert spangroup[2].label_ == 'GreatCity'
            assert_almost_equal(0.9, spangroup.attrs['scores'][2], 5)
        else:
            assert spangroup[2].label_ == 'City'
            assert_almost_equal(0.7, spangroup.attrs['scores'][2], 5)
    assert spangroup[-1].text == 'Greater London'
    assert spangroup[-1].label_ == 'GreatCity'
    assert_almost_equal(0.9, spangroup.attrs['scores'][-1], 5)