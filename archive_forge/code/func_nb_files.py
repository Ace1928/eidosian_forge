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
def nb_files(self) -> int:
    """
        (property) Total number of files in the revision.
        """
    return len(self.files)