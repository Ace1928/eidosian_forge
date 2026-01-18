import random
import warnings
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import srsly
from thinc.api import CosineDistance, Model, Optimizer, set_dropout_rate
from thinc.types import Floats2d
from ... import util
from ...errors import Errors, Warnings
from ...kb import Candidate, KnowledgeBase
from ...language import Language
from ...ml import empty_kb
from ...scorer import Scorer
from ...tokens import Doc, Span
from ...training import Example, validate_examples, validate_get_examples
from ...util import SimpleFrozenList
from ...vocab import Vocab
from ..pipe import deserialize_config
from ..trainable_pipe import TrainablePipe
def set_kb(self, kb_loader: Callable[[Vocab], KnowledgeBase]):
    """Define the KB of this pipe by providing a function that will
        create it using this object's vocab."""
    if not callable(kb_loader):
        raise ValueError(Errors.E885.format(arg_type=type(kb_loader)))
    self.kb = kb_loader(self.vocab)