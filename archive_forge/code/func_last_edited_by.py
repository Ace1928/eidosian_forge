from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Union
from .constants import REPO_TYPE_MODEL
from .utils import parse_datetime
@property
def last_edited_by(self) -> str:
    """The last edit time, as a `datetime` object."""
    return self._event['data']['latest'].get('author', {}).get('name', 'deleted')