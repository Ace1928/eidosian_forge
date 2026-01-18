from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Union
from .constants import REPO_TYPE_MODEL
from .utils import parse_datetime
@property
def edit_history(self) -> List[dict]:
    """The edit history of the comment"""
    return self._event['data']['history']