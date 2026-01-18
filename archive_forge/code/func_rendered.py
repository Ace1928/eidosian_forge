from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Union
from .constants import REPO_TYPE_MODEL
from .utils import parse_datetime
@property
def rendered(self) -> str:
    """The rendered comment, as a HTML string"""
    return self._event['data']['latest']['html']