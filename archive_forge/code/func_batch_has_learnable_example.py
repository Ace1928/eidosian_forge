import random
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import srsly
from thinc.api import Config, CosineDistance, Model, Optimizer, set_dropout_rate
from thinc.types import Floats2d
from .. import util
from ..errors import Errors
from ..kb import Candidate, KnowledgeBase
from ..language import Language
from ..ml import empty_kb
from ..scorer import Scorer
from ..tokens import Doc, Span
from ..training import Example, validate_examples, validate_get_examples
from ..util import SimpleFrozenList, registry
from ..vocab import Vocab
from .legacy.entity_linker import EntityLinker_v1
from .pipe import deserialize_config
from .trainable_pipe import TrainablePipe
def batch_has_learnable_example(self, examples):
    """Check if a batch contains a learnable example.

        If one isn't present, then the update step needs to be skipped.
        """
    for eg in examples:
        for ent in eg.predicted.ents:
            candidates = list(self.get_candidates(self.kb, ent))
            if candidates:
                return True
    return False