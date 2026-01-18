from typing import Callable, Iterable, Iterator
import pytest
from thinc.api import Config
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example
from spacy.training.loop import train
from spacy.util import load_model_from_config, registry
@Language.factory('assert_sents', default_config={})
def assert_sents(nlp, name):
    return AssertSents(name)