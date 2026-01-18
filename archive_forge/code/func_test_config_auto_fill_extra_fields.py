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
def test_config_auto_fill_extra_fields():
    config = Config({'nlp': {'lang': 'en'}, 'training': {}})
    assert load_model_from_config(config, auto_fill=True)
    config = Config({'nlp': {'lang': 'en'}, 'training': {'extra': 'hello'}})
    nlp = load_model_from_config(config, auto_fill=True, validate=False)
    assert 'extra' not in nlp.config['training']
    load_model_from_config(nlp.config)