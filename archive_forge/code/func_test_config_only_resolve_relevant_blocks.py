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
def test_config_only_resolve_relevant_blocks():
    """Test that only the relevant blocks are resolved in the different methods
    and that invalid blocks are ignored if needed. For instance, the [initialize]
    shouldn't be resolved at runtime.
    """
    nlp = English()
    config = nlp.config
    config['training']['before_to_disk'] = {'@misc': 'nonexistent'}
    config['initialize']['lookups'] = {'@misc': 'nonexistent'}
    nlp = load_model_from_config(config, auto_fill=True)
    with pytest.raises(RegistryError):
        nlp.initialize()
    nlp.config['initialize']['lookups'] = None
    nlp.initialize()