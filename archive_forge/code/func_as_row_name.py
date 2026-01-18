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
def as_row_name(self) -> str:
    return self.sub_label or self.stmt or '[Unknown]'