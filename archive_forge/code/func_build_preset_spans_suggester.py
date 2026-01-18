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
@registry.misc('spacy.preset_spans_suggester.v1')
def build_preset_spans_suggester(spans_key: str) -> Suggester:
    """Suggest all spans that are already stored in doc.spans[spans_key].
    This is useful when an upstream component is used to set the spans
    on the Doc such as a SpanRuler or SpanFinder."""
    return partial(preset_spans_suggester, spans_key=spans_key)