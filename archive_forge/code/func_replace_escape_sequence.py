import io as StringIO
import re
from typing import Dict, Iterable, List, Match, Optional, TextIO, Tuple
from .metrics_core import Metric
from .samples import Sample
def replace_escape_sequence(match: Match[str]) -> str:
    return ESCAPE_SEQUENCES[match.group(0)]