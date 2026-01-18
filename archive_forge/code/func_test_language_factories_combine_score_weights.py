import pytest
from thinc.api import ConfigValidationError, Linear, Model
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.util import SimpleFrozenDict, combine_score_weights, registry
from ..util import make_tempdir
@pytest.mark.parametrize('weights,override,expected', [([{'a': 1.0}, {'b': 1.0}, {'c': 1.0}], {}, {'a': 0.33, 'b': 0.33, 'c': 0.33}), ([{'a': 1.0}, {'b': 50}, {'c': 100}], {}, {'a': 0.01, 'b': 0.33, 'c': 0.66}), ([{'a': 0.7, 'b': 0.3}, {'c': 1.0}, {'d': 0.5, 'e': 0.5}], {}, {'a': 0.23, 'b': 0.1, 'c': 0.33, 'd': 0.17, 'e': 0.17}), ([{'a': 100, 'b': 300}, {'c': 50, 'd': 50}], {}, {'a': 0.2, 'b': 0.6, 'c': 0.1, 'd': 0.1}), ([{'a': 0.5, 'b': 0.5}, {'b': 1.0}], {}, {'a': 0.33, 'b': 0.67}), ([{'a': 0.5, 'b': 0.0}], {}, {'a': 1.0, 'b': 0.0}), ([{'a': 0.5, 'b': 0.5}, {'b': 1.0}], {'a': 0.0}, {'a': 0.0, 'b': 1.0}), ([{'a': 0.0, 'b': 0.0}, {'c': 0.0}], {}, {'a': 0.0, 'b': 0.0, 'c': 0.0}), ([{'a': 0.0, 'b': 0.0}, {'c': 1.0}], {}, {'a': 0.0, 'b': 0.0, 'c': 1.0}), ([{'a': 0.0, 'b': 0.0}, {'c': 0.0}], {'c': 0.2}, {'a': 0.0, 'b': 0.0, 'c': 1.0}), ([{'a': 0.5, 'b': 0.5, 'c': 1.0, 'd': 1.0}], {'a': 0.0, 'b': 0.0}, {'a': 0.0, 'b': 0.0, 'c': 0.5, 'd': 0.5}), ([{'a': 0.5, 'b': 0.5, 'c': 1.0, 'd': 1.0}], {'a': 0.0, 'b': 0.0, 'f': 0.0}, {'a': 0.0, 'b': 0.0, 'c': 0.5, 'd': 0.5, 'f': 0.0})])
def test_language_factories_combine_score_weights(weights, override, expected):
    result = combine_score_weights(weights, override)
    assert sum(result.values()) in (0.99, 1.0, 0.0)
    assert result == expected