import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
def add_weakref_references():
    if type(obj) is weakref.ref:
        referents = gc.get_referents(obj)
        if len(referents) == 1:
            target = referents[0]
            add_reference('__callback__', target)