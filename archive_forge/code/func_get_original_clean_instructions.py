import __future__
import ast
import dis
import inspect
import io
import linecache
import re
import sys
import types
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import islice
from itertools import zip_longest
from operator import attrgetter
from pathlib import Path
from threading import RLock
from tokenize import detect_encoding
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Sized, Tuple, \
def get_original_clean_instructions(self):
    result = self.clean_instructions(self.code)
    if not any((inst.opname == 'JUMP_IF_NOT_DEBUG' for inst in self.compile_instructions())):
        result = [inst for inst in result if inst.opname != 'JUMP_IF_NOT_DEBUG']
    return result