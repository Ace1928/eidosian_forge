import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union
from ..constants import HF_HUB_CACHE
from . import logging
@property
def last_accessed_str(self) -> str:
    """
        (property) Last time a blob file of the repo has been accessed, returned as a
        human-readable string.

        Example: "2 weeks ago".
        """
    return _format_timesince(self.last_accessed)