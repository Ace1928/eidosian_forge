import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union
@property
def ignore_llm(self) -> bool:
    """Whether to ignore LLM callbacks."""
    return self.ignore_llm_