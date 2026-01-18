import collections
import contextlib
import dataclasses
import os
import shutil
import tempfile
import textwrap
import time
from typing import cast, Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple
import uuid
import torch
def meets_confidence(self, threshold: float=_IQR_WARN_THRESHOLD) -> bool:
    return self.iqr / self.median < threshold