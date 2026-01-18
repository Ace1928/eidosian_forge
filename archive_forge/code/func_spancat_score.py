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
def spancat_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    kwargs = dict(kwargs)
    attr_prefix = 'spans_'
    key = kwargs['spans_key']
    kwargs.setdefault('attr', f'{attr_prefix}{key}')
    kwargs.setdefault('allow_overlap', True)
    kwargs.setdefault('getter', lambda doc, key: doc.spans.get(key[len(attr_prefix):], []))
    kwargs.setdefault('has_annotation', lambda doc: key in doc.spans)
    return Scorer.score_spans(examples, **kwargs)