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
@property
def has_warnings(self) -> bool:
    self._lazy_init()
    return bool(self._warnings)