from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple
from thinc.api import (
from thinc.types import Floats2d
from ...errors import Errors
from ...kb import (
from ...tokens import Doc, Span
from ...util import registry
from ...vocab import Vocab
from ..extract_spans import extract_spans
def empty_kb_factory(vocab: Vocab):
    return InMemoryLookupKB(vocab=vocab, entity_vector_length=entity_vector_length)