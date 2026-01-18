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
def test_make_spangroup_negative_label():
    fix_random_seed(0)
    nlp_single = Language()
    nlp_multi = Language()
    spancat_single = nlp_single.add_pipe('spancat', config={'spans_key': SPAN_KEY, 'threshold': 0.1, 'max_positive': 1})
    spancat_multi = nlp_multi.add_pipe('spancat', config={'spans_key': SPAN_KEY, 'threshold': 0.1, 'max_positive': 2})
    spancat_single.add_negative_label = True
    spancat_multi.add_negative_label = True
    doc = nlp_single.make_doc('Greater London')
    labels = ['Thing', 'City', 'Person', 'GreatCity']
    for label in labels:
        spancat_multi.add_label(label)
        spancat_single.add_label(label)
    ngram_suggester = registry.misc.get('spacy.ngram_suggester.v1')(sizes=[1, 2])
    indices = ngram_suggester([doc])[0].dataXd
    assert_array_equal(OPS.to_numpy(indices), numpy.asarray([[0, 1], [1, 2], [0, 2]]))
    scores = numpy.asarray([[0.2, 0.4, 0.3, 0.1, 0.1], [0.1, 0.6, 0.2, 0.4, 0.9], [0.8, 0.7, 0.3, 0.9, 0.1]], dtype='f')
    spangroup_multi = spancat_multi._make_span_group_multilabel(doc, indices, scores)
    spangroup_single = spancat_single._make_span_group_singlelabel(doc, indices, scores)
    assert len(spangroup_single) == 2
    assert spangroup_single[0].text == 'Greater'
    assert spangroup_single[0].label_ == 'City'
    assert_almost_equal(0.4, spangroup_single.attrs['scores'][0], 5)
    assert spangroup_single[1].text == 'Greater London'
    assert spangroup_single[1].label_ == 'GreatCity'
    assert spangroup_single.attrs['scores'][1] == 0.9
    assert_almost_equal(0.9, spangroup_single.attrs['scores'][1], 5)
    assert len(spangroup_multi) == 6
    assert spangroup_multi[0].text == 'Greater'
    assert spangroup_multi[0].label_ == 'City'
    assert_almost_equal(0.4, spangroup_multi.attrs['scores'][0], 5)
    assert spangroup_multi[1].text == 'Greater'
    assert spangroup_multi[1].label_ == 'Person'
    assert_almost_equal(0.3, spangroup_multi.attrs['scores'][1], 5)
    assert spangroup_multi[2].text == 'London'
    assert spangroup_multi[2].label_ == 'City'
    assert_almost_equal(0.6, spangroup_multi.attrs['scores'][2], 5)
    assert spangroup_multi[3].text == 'London'
    assert spangroup_multi[3].label_ == 'GreatCity'
    assert_almost_equal(0.4, spangroup_multi.attrs['scores'][3], 5)
    assert spangroup_multi[4].text == 'Greater London'
    assert spangroup_multi[4].label_ == 'Thing'
    assert spangroup_multi[4].text == 'Greater London'
    assert_almost_equal(0.8, spangroup_multi.attrs['scores'][4], 5)
    assert spangroup_multi[5].text == 'Greater London'
    assert spangroup_multi[5].label_ == 'GreatCity'
    assert_almost_equal(0.9, spangroup_multi.attrs['scores'][5], 5)