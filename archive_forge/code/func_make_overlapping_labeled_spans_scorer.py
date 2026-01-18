import warnings
from functools import partial
from pathlib import Path
from typing import (
import srsly
from .. import util
from ..errors import Errors, Warnings
from ..language import Language
from ..matcher import Matcher, PhraseMatcher
from ..matcher.levenshtein import levenshtein_compare
from ..scorer import Scorer
from ..tokens import Doc, Span
from ..training import Example
from ..util import SimpleFrozenList, ensure_path, registry
from .pipe import Pipe
@registry.scorers('spacy.overlapping_labeled_spans_scorer.v1')
def make_overlapping_labeled_spans_scorer(spans_key: str=DEFAULT_SPANS_KEY):
    return partial(overlapping_labeled_spans_score, spans_key=spans_key)