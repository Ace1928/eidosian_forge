from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy
from thinc.api import Config, Model, Optimizer, get_array_module, set_dropout_rate
from thinc.types import Floats2d
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc
from ..training import Example, validate_examples, validate_get_examples
from ..util import registry
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
@registry.scorers('spacy.textcat_scorer.v2')
def make_textcat_scorer():
    return textcat_score