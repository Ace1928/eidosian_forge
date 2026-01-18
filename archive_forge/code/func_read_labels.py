import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Optional, Union
import srsly
from .. import util
from ..errors import Errors, Warnings
from ..tokens import Doc, DocBin
from ..vocab import Vocab
from .augment import dont_augment
from .example import Example
@util.registry.readers('spacy.read_labels.v1')
def read_labels(path: Path, *, require: bool=False):
    if not require and (not path.exists()):
        return None
    return srsly.read_json(path)