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
@util.registry.readers('spacy.Corpus.v1')
def create_docbin_reader(path: Optional[Path], gold_preproc: bool, max_length: int=0, limit: int=0, augmenter: Optional[Callable]=None) -> Callable[['Language'], Iterable[Example]]:
    if path is None:
        raise ValueError(Errors.E913)
    util.logger.debug('Loading corpus from path: %s', path)
    return Corpus(path, gold_preproc=gold_preproc, max_length=max_length, limit=limit, augmenter=augmenter)