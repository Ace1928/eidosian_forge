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
def expected_freed_size_str(self) -> str:
    """
        (property) Expected size that will be freed as a human-readable string.

        Example: "42.2K".
        """
    return _format_size(self.expected_freed_size)