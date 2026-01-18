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
@pytest.mark.skip(reason='Test is unreliable for unknown reason')
def test_doc_gc():
    nlp = Language()
    spancat = nlp.add_pipe('spancat', config={'spans_key': SPAN_KEY})
    spancat.add_label('PERSON')
    nlp.initialize()
    texts = ['Just a sentence.', 'I like London and Berlin', 'I like Berlin', 'I eat ham.']
    all_spans = [doc.spans for doc in nlp.pipe(texts)]
    for text, spangroups in zip(texts, all_spans):
        assert isinstance(spangroups, SpanGroups)
        for key, spangroup in spangroups.items():
            assert isinstance(spangroup, SpanGroup)
            assert len(spangroup) > 0
            with pytest.raises(RuntimeError):
                spangroup[0]