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
def preset_spans_suggester(docs: Iterable[Doc], spans_key: str, *, ops: Optional[Ops]=None) -> Ragged:
    if ops is None:
        ops = get_current_ops()
    spans = []
    lengths = []
    for doc in docs:
        length = 0
        if doc.spans[spans_key]:
            for span in doc.spans[spans_key]:
                spans.append([span.start, span.end])
                length += 1
        lengths.append(length)
    lengths_array = cast(Ints1d, ops.asarray(lengths, dtype='i'))
    if len(spans) > 0:
        output = Ragged(ops.asarray(spans, dtype='i'), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype='i'), lengths_array)
    return output