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
def test_config_nlp_roundtrip():
    """Test that a config produced by the nlp object passes training config
    validation."""
    nlp = English()
    nlp.add_pipe('entity_ruler')
    nlp.add_pipe('ner')
    new_nlp = load_model_from_config(nlp.config, auto_fill=False)
    assert new_nlp.config == nlp.config
    assert new_nlp.pipe_names == nlp.pipe_names
    assert new_nlp._pipe_configs == nlp._pipe_configs
    assert new_nlp._pipe_meta == nlp._pipe_meta
    assert new_nlp._factory_meta == nlp._factory_meta