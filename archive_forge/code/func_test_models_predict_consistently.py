from typing import List
import numpy
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from thinc.api import (
from spacy.lang.en import English
from spacy.lang.en.examples import sentences as EN_SENTENCES
from spacy.ml.extract_spans import _get_span_indices, extract_spans
from spacy.ml.models import (
from spacy.ml.staticvectors import StaticVectors
from spacy.util import registry
@pytest.mark.parametrize('seed,model_func,kwargs,get_X', [(0, build_Tok2Vec_model, get_tok2vec_kwargs(), get_docs), (0, build_bow_text_classifier, get_textcat_bow_kwargs(), get_docs), (0, build_simple_cnn_text_classifier, get_textcat_cnn_kwargs(), get_docs)])
def test_models_predict_consistently(seed, model_func, kwargs, get_X):
    fix_random_seed(seed)
    model1 = model_func(**kwargs).initialize()
    Y1 = model1.predict(get_X())
    fix_random_seed(seed)
    model2 = model_func(**kwargs).initialize()
    Y2 = model2.predict(get_X())
    if model1.has_ref('tok2vec'):
        tok2vec1 = model1.get_ref('tok2vec').predict(get_X())
        tok2vec2 = model2.get_ref('tok2vec').predict(get_X())
        for i in range(len(tok2vec1)):
            for j in range(len(tok2vec1[i])):
                assert_array_equal(numpy.asarray(model1.ops.to_numpy(tok2vec1[i][j])), numpy.asarray(model2.ops.to_numpy(tok2vec2[i][j])))
    try:
        Y1 = model1.ops.to_numpy(Y1)
        Y2 = model2.ops.to_numpy(Y2)
    except Exception:
        pass
    if isinstance(Y1, numpy.ndarray):
        assert_array_equal(Y1, Y2)
    elif isinstance(Y1, List):
        assert len(Y1) == len(Y2)
        for y1, y2 in zip(Y1, Y2):
            try:
                y1 = model1.ops.to_numpy(y1)
                y2 = model2.ops.to_numpy(y2)
            except Exception:
                pass
            assert_array_equal(y1, y2)
    else:
        raise ValueError(f'Could not compare type {type(Y1)}')