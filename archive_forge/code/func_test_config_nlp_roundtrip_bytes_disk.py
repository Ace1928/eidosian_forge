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
def test_config_nlp_roundtrip_bytes_disk():
    """Test that the config is serialized correctly and not interpolated
    by mistake."""
    nlp = English()
    nlp_bytes = nlp.to_bytes()
    new_nlp = English().from_bytes(nlp_bytes)
    assert new_nlp.config == nlp.config
    nlp = English()
    with make_tempdir() as d:
        nlp.to_disk(d)
        new_nlp = spacy.load(d)
    assert new_nlp.config == nlp.config