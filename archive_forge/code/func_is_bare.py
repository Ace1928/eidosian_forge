from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from urllib.parse import urlparse
def is_bare(self) -> bool:
    return self.host is None