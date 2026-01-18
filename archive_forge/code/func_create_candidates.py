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
@registry.misc('spacy.CandidateGenerator.v1')
def create_candidates() -> Callable[[KnowledgeBase, Span], Iterable[Candidate]]:
    return get_candidates