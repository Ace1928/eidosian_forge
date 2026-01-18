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
@registry.misc('spacy.ngram_range_suggester.v1')
def build_ngram_range_suggester(min_size: int, max_size: int) -> Suggester:
    """Suggest all spans of the given lengths between a given min and max value - both inclusive.
    Spans are returned as a ragged array of integers. The array has two columns,
    indicating the start and end position."""
    sizes = list(range(min_size, max_size + 1))
    return build_ngram_suggester(sizes)