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
@Language.factory('spancat', assigns=['doc.spans'], default_config={'threshold': 0.5, 'spans_key': DEFAULT_SPANS_KEY, 'max_positive': None, 'model': DEFAULT_SPANCAT_MODEL, 'suggester': {'@misc': 'spacy.ngram_suggester.v1', 'sizes': [1, 2, 3]}, 'scorer': {'@scorers': 'spacy.spancat_scorer.v1'}}, default_score_weights={'spans_sc_f': 1.0, 'spans_sc_p': 0.0, 'spans_sc_r': 0.0})
def make_spancat(nlp: Language, name: str, suggester: Suggester, model: Model[Tuple[List[Doc], Ragged], Floats2d], spans_key: str, scorer: Optional[Callable], threshold: float, max_positive: Optional[int]) -> 'SpanCategorizer':
    """Create a SpanCategorizer component and configure it for multi-label
    classification to be able to assign multiple labels for each span.
    The span categorizer consists of two
    parts: a suggester function that proposes candidate spans, and a labeller
    model that predicts one or more labels for each span.

    name (str): The component instance name, used to add entries to the
        losses during training.
    suggester (Callable[[Iterable[Doc], Optional[Ops]], Ragged]): A function that suggests spans.
        Spans are returned as a ragged array with two integer columns, for the
        start and end positions.
    model (Model[Tuple[List[Doc], Ragged], Floats2d]): A model instance that
        is given a list of documents and (start, end) indices representing
        candidate span offsets. The model predicts a probability for each category
        for each span.
    spans_key (str): Key of the doc.spans dict to save the spans under. During
        initialization and training, the component will look for spans on the
        reference document under the same key.
    scorer (Optional[Callable]): The scoring method. Defaults to
        Scorer.score_spans for the Doc.spans[spans_key] with overlapping
        spans allowed.
    threshold (float): Minimum probability to consider a prediction positive.
        Spans with a positive prediction will be saved on the Doc. Defaults to
        0.5.
    max_positive (Optional[int]): Maximum number of labels to consider positive
        per span. Defaults to None, indicating no limit.
    """
    return SpanCategorizer(nlp.vocab, model=model, suggester=suggester, name=name, spans_key=spans_key, negative_weight=None, allow_overlap=True, max_positive=max_positive, threshold=threshold, scorer=scorer, add_negative_label=False)