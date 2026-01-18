import pytest
from catalogue import RegistryError
from thinc.api import Config, ConfigValidationError
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import DEFAULT_CONFIG, DEFAULT_CONFIG_PRETRAIN_PATH, Language
from spacy.ml.models import (
from spacy.schemas import ConfigSchema, ConfigSchemaPretrain
from spacy.training import Example
from spacy.util import (
from ..util import make_tempdir
def test_create_nlp_from_config_multiple_instances():
    """Test that the nlp object is created correctly for a config with multiple
    instances of the same component."""
    config = Config().from_str(nlp_config_string)
    config['components'] = {'t2v': config['components']['tok2vec'], 'tagger1': config['components']['tagger'], 'tagger2': config['components']['tagger']}
    config['nlp']['pipeline'] = list(config['components'].keys())
    nlp = load_model_from_config(config, auto_fill=True)
    assert nlp.pipe_names == ['t2v', 'tagger1', 'tagger2']
    assert nlp.get_pipe_meta('t2v').factory == 'tok2vec'
    assert nlp.get_pipe_meta('tagger1').factory == 'tagger'
    assert nlp.get_pipe_meta('tagger2').factory == 'tagger'
    pipeline_config = nlp.config['components']
    assert len(pipeline_config) == 3
    assert list(pipeline_config.keys()) == ['t2v', 'tagger1', 'tagger2']
    assert nlp.config['nlp']['pipeline'] == ['t2v', 'tagger1', 'tagger2']