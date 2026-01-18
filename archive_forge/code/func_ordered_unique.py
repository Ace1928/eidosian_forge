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
def ordered_unique(elements: Iterable[Any]) -> List[Any]:
    return list(collections.OrderedDict({i: None for i in elements}).keys())