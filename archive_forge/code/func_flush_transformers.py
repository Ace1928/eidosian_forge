from __future__ import annotations
from warnings import warn
import ast
import codeop
import io
import re
import sys
import tokenize
import warnings
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from types import CodeType
from IPython.core.inputtransformer import (leading_indent,
from IPython.utils import tokenutil
from IPython.core.inputtransformer import (ESC_SHELL, ESC_SH_CAP, ESC_HELP,
def flush_transformers(self: Self):

    def _flush(transform, outs: List[str]):
        """yield transformed lines

            always strings, never None

            transform: the current transform
            outs: an iterable of previously transformed inputs.
                 Each may be multiline, which will be passed
                 one line at a time to transform.
            """
        for out in outs:
            for line in out.splitlines():
                tmp = transform.push(line)
                if tmp is not None:
                    yield tmp
        tmp = transform.reset()
        if tmp is not None:
            yield tmp
    out: List[str] = []
    for t in self.transforms_in_use:
        out = _flush(t, out)
    out = list(out)
    if out:
        self._store('\n'.join(out))