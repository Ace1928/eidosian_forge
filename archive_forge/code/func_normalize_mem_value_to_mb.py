from typing import Dict, Any, Optional, Tuple, IO
import re
import sys
from spacy import Language
from .util import LoggerT
def normalize_mem_value_to_mb(name: str, value: int) -> Tuple[str, float]:
    if '_bytes' in name:
        return (re.sub('_bytes', '_megabytes', name), value / 1024.0 ** 2)
    else:
        return (name, value)