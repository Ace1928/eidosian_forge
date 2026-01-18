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
def test_ngram_suggester(en_tokenizer):
    for size in [1, 2, 3]:
        ngram_suggester = registry.misc.get('spacy.ngram_suggester.v1')(sizes=[size])
        docs = [en_tokenizer(text) for text in ['a', 'a b', 'a b c', 'a b c d', 'a b c d e', 'a ' * 100]]
        ngrams = ngram_suggester(docs)
        for s in ngrams.data:
            assert s[1] - s[0] == size
        offset = 0
        for i, doc in enumerate(docs):
            spans = ngrams.dataXd[offset:offset + ngrams.lengths[i]]
            spans_set = set()
            for span in spans:
                assert 0 <= span[0] < len(doc)
                assert 0 < span[1] <= len(doc)
                spans_set.add((int(span[0]), int(span[1])))
            assert spans.shape[0] == len(spans_set)
            offset += ngrams.lengths[i]
        assert_array_equal(OPS.to_numpy(ngrams.lengths), [max(0, len(doc) - (size - 1)) for doc in docs])
    ngram_suggester = registry.misc.get('spacy.ngram_suggester.v1')(sizes=[1, 2, 3])
    docs = [en_tokenizer(text) for text in ['a', 'a b', 'a b c', 'a b c d', 'a b c d e']]
    ngrams = ngram_suggester(docs)
    assert_array_equal(OPS.to_numpy(ngrams.lengths), [1, 3, 6, 9, 12])
    assert_array_equal(OPS.to_numpy(ngrams.data), [[0, 1], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [0, 3], [0, 1], [1, 2], [2, 3], [3, 4], [0, 2], [1, 3], [2, 4], [0, 3], [1, 4], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 2], [1, 3], [2, 4], [3, 5], [0, 3], [1, 4], [2, 5]])
    ngram_suggester = registry.misc.get('spacy.ngram_suggester.v1')(sizes=[1])
    docs = [en_tokenizer(text) for text in ['', 'a', '']]
    ngrams = ngram_suggester(docs)
    assert_array_equal(OPS.to_numpy(ngrams.lengths), [len(doc) for doc in docs])
    ngram_suggester = registry.misc.get('spacy.ngram_suggester.v1')(sizes=[1])
    docs = [en_tokenizer(text) for text in ['', '', '']]
    ngrams = ngram_suggester(docs)
    assert_array_equal(OPS.to_numpy(ngrams.lengths), [len(doc) for doc in docs])