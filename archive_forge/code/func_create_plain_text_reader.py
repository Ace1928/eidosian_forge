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
@util.registry.readers('spacy.PlainTextCorpus.v1')
def create_plain_text_reader(path: Optional[Path], min_length: int=0, max_length: int=0) -> Callable[['Language'], Iterable[Example]]:
    """Iterate Example objects from a file or directory of plain text
    UTF-8 files with one line per doc.

    path (Path): The directory or filename to read from.
    min_length (int): Minimum document length (in tokens). Shorter documents
        will be skipped. Defaults to 0, which indicates no limit.
    max_length (int): Maximum document length (in tokens). Longer documents will
        be skipped. Defaults to 0, which indicates no limit.

    DOCS: https://spacy.io/api/corpus#plaintextcorpus
    """
    if path is None:
        raise ValueError(Errors.E913)
    return PlainTextCorpus(path, min_length=min_length, max_length=max_length)