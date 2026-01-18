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
def make_examples_gold_preproc(self, nlp: 'Language', reference_docs: Iterable[Doc]) -> Iterator[Example]:
    for reference in reference_docs:
        if reference.has_annotation('SENT_START'):
            ref_sents = [sent.as_doc() for sent in reference.sents]
        else:
            ref_sents = [reference]
        for ref_sent in ref_sents:
            eg = self._make_example(nlp, ref_sent, True)
            if len(eg.x):
                yield eg