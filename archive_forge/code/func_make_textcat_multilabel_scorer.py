from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional
from thinc.api import Config, Model
from thinc.types import Floats2d
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc
from ..training import Example, validate_get_examples
from ..util import registry
from ..vocab import Vocab
from .textcat import TextCategorizer
@registry.scorers('spacy.textcat_multilabel_scorer.v2')
def make_textcat_multilabel_scorer():
    return textcat_multilabel_score