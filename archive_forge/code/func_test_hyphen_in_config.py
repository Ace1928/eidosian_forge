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
def test_hyphen_in_config():
    hyphen_config_str = '\n    [nlp]\n    lang = "en"\n    pipeline = ["my_punctual_component"]\n\n    [components]\n\n    [components.my_punctual_component]\n    factory = "my_punctual_component"\n    punctuation = ["?","-"]\n    '

    @spacy.Language.factory('my_punctual_component')
    class MyPunctualComponent(object):
        name = 'my_punctual_component'

        def __init__(self, nlp, name, punctuation):
            self.punctuation = punctuation
    nlp = English.from_config(load_config_from_str(hyphen_config_str))
    assert nlp.get_pipe('my_punctual_component').punctuation == ['?', '-']