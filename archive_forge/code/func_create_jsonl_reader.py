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
@util.registry.readers('spacy.JsonlCorpus.v1')
def create_jsonl_reader(path: Optional[Union[str, Path]], min_length: int=0, max_length: int=0, limit: int=0) -> Callable[['Language'], Iterable[Example]]:
    return JsonlCorpus(path, min_length=min_length, max_length=max_length, limit=limit)