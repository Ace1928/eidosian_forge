from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import srsly
from .. import util
from ..errors import Errors
from ..language import Language
from ..matcher import Matcher
from ..scorer import Scorer
from ..symbols import IDS
from ..tokens import Doc, Span
from ..tokens._retokenize import normalize_token_attrs, set_token_attrs
from ..training import Example
from ..util import SimpleFrozenList, registry
from ..vocab import Vocab
from .pipe import Pipe
def morph_key_getter(token, attr):
    return getattr(token, attr).key