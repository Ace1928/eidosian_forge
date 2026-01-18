from pathlib import Path
from typing import Any, Callable, Dict, Iterable
import srsly
from numpy import zeros
from thinc.api import Config
from spacy import Errors, util
from spacy.kb.kb_in_memory import InMemoryLookupKB
from spacy.util import SimpleFrozenList, ensure_path, load_model_from_config, registry
from spacy.vocab import Vocab
from ..util import make_tempdir
def custom_kb_factory(vocab):
    kb = SubInMemoryLookupKB(vocab=vocab, entity_vector_length=entity_vector_length, custom_field=custom_field)
    kb.add_entity('random_entity', 0.0, zeros(entity_vector_length))
    return kb