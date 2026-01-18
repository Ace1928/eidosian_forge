import os
import sys
import time
import traceback
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Sequence
from sphinx.errors import SphinxParallelError
from sphinx.util import logging
def make_chunks(arguments: Sequence[str], nproc: int, maxbatch: int=10) -> List[Any]:
    nargs = len(arguments)
    chunksize = nargs // nproc
    if chunksize >= maxbatch:
        chunksize = int(sqrt(nargs / nproc * maxbatch))
    if chunksize == 0:
        chunksize = 1
    nchunks, rest = divmod(nargs, chunksize)
    if rest:
        nchunks += 1
    return [arguments[i * chunksize:(i + 1) * chunksize] for i in range(nchunks)]