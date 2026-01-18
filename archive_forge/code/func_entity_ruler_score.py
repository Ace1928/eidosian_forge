import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import srsly
from ..errors import Errors, Warnings
from ..language import Language
from ..matcher import Matcher, PhraseMatcher
from ..matcher.levenshtein import levenshtein_compare
from ..scorer import get_ner_prf
from ..tokens import Doc, Span
from ..training import Example
from ..util import SimpleFrozenList, ensure_path, from_disk, registry, to_disk
from .pipe import Pipe
def entity_ruler_score(examples, **kwargs):
    return get_ner_prf(examples)