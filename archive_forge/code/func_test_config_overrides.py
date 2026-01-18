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
def test_config_overrides():
    overrides_nested = {'nlp': {'lang': 'de', 'pipeline': ['tagger']}}
    overrides_dot = {'nlp.lang': 'de', 'nlp.pipeline': ['tagger']}
    config = Config().from_str(nlp_config_string, overrides=overrides_dot)
    nlp = load_model_from_config(config, auto_fill=True)
    assert isinstance(nlp, German)
    assert nlp.pipe_names == ['tagger']
    base_config = Config().from_str(nlp_config_string)
    base_nlp = load_model_from_config(base_config, auto_fill=True)
    assert isinstance(base_nlp, English)
    assert base_nlp.pipe_names == ['tok2vec', 'tagger']
    with make_tempdir() as d:
        base_nlp.to_disk(d)
        nlp = spacy.load(d, config=overrides_nested)
    assert isinstance(nlp, German)
    assert nlp.pipe_names == ['tagger']
    with make_tempdir() as d:
        base_nlp.to_disk(d)
        nlp = spacy.load(d, config=overrides_dot)
    assert isinstance(nlp, German)
    assert nlp.pipe_names == ['tagger']
    with make_tempdir() as d:
        base_nlp.to_disk(d)
        nlp = spacy.load(d)
    assert isinstance(nlp, English)
    assert nlp.pipe_names == ['tok2vec', 'tagger']