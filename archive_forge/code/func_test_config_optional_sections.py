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
def test_config_optional_sections():
    config = Config().from_str(nlp_config_string)
    config = DEFAULT_CONFIG.merge(config)
    assert 'pretraining' not in config
    filled = registry.fill(config, schema=ConfigSchema, validate=False)
    new_config = Config().from_str(filled.to_str())
    assert new_config['pretraining'] == {}