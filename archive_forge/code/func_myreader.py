from typing import Callable, Dict, Iterable
import pytest
from thinc.api import Config, fix_random_seed
from spacy import Language
from spacy.schemas import ConfigSchemaTraining
from spacy.training import Example
from spacy.util import load_model_from_config, registry, resolve_dot_names
@registry.readers('myreader.v1')
def myreader() -> Dict[str, Callable[[Language], Iterable[Example]]]:
    annots = {'cats': {'POS': 1.0, 'NEG': 0.0}}

    def reader(nlp: Language):
        doc = nlp.make_doc(f'This is an example')
        return [Example.from_dict(doc, annots)]
    return {'train': reader, 'dev': reader, 'extra': reader, 'something': reader}