from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import numpy
from thinc.api import Config, Model, Ops, Optimizer, get_current_ops, set_dropout_rate
from thinc.types import Floats2d, Ints1d, Ints2d, Ragged
from ..compat import Protocol, runtime_checkable
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc, Span, SpanGroup
from ..training import Example, validate_examples
from ..util import registry
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
def set_candidates(self, docs: Iterable[Doc], *, candidates_key: str='candidates') -> None:
    """Use the spancat suggester to add a list of span candidates to a list of docs.
        This method is intended to be used for debugging purposes.

        docs (Iterable[Doc]): The documents to modify.
        candidates_key (str): Key of the Doc.spans dict to save the candidate spans under.

        DOCS: https://spacy.io/api/spancategorizer#set_candidates
        """
    suggester_output = self.suggester(docs, ops=self.model.ops)
    for candidates, doc in zip(suggester_output, docs):
        doc.spans[candidates_key] = []
        for index in candidates.dataXd:
            doc.spans[candidates_key].append(doc[index[0]:index[1]])