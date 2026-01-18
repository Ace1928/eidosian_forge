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
def test_language_factories_scores():
    name = 'test_language_factories_scores'
    func = lambda nlp, name: lambda doc: doc
    weights1 = {'a1': 0.5, 'a2': 0.5}
    weights2 = {'b1': 0.2, 'b2': 0.7, 'b3': 0.1}
    Language.factory(f'{name}1', default_score_weights=weights1, func=func)
    Language.factory(f'{name}2', default_score_weights=weights2, func=func)
    meta1 = Language.get_factory_meta(f'{name}1')
    assert meta1.default_score_weights == weights1
    meta2 = Language.get_factory_meta(f'{name}2')
    assert meta2.default_score_weights == weights2
    nlp = Language()
    nlp._config['training']['score_weights'] = {}
    nlp.add_pipe(f'{name}1')
    nlp.add_pipe(f'{name}2')
    cfg = nlp.config['training']
    expected_weights = {'a1': 0.25, 'a2': 0.25, 'b1': 0.1, 'b2': 0.35, 'b3': 0.05}
    assert cfg['score_weights'] == expected_weights
    config = nlp.config.copy()
    config['training']['score_weights']['a1'] = 0.0
    config['training']['score_weights']['b3'] = 1.3
    nlp = English.from_config(config)
    score_weights = nlp.config['training']['score_weights']
    expected = {'a1': 0.0, 'a2': 0.12, 'b1': 0.05, 'b2': 0.17, 'b3': 0.65}
    assert score_weights == expected
    config = nlp.config.copy()
    config['training']['score_weights']['a1'] = None
    nlp = English.from_config(config)
    score_weights = nlp.config['training']['score_weights']
    expected = {'a1': None, 'a2': 0.12, 'b1': 0.05, 'b2': 0.17, 'b3': 0.66}
    assert score_weights == expected