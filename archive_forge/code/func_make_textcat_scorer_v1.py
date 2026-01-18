from typing import Any, Callable, Dict, Iterable, Optional
from spacy.scorer import PRFScore, ROCAUCScore
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import SimpleFrozenList
def make_textcat_scorer_v1():
    return textcat_score_v1