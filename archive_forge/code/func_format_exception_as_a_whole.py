from collections.abc import Sequence
import functools
import inspect
import linecache
import pydoc
import sys
import time
import traceback
import types
from types import TracebackType
from typing import Any, List, Optional, Tuple
import stack_data
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.styles import get_style_by_name
import IPython.utils.colorable as colorable
from IPython import get_ipython
from IPython.core import debugger
from IPython.core.display_trap import DisplayTrap
from IPython.core.excolors import exception_colors
from IPython.utils import PyColorize
from IPython.utils import path as util_path
from IPython.utils import py3compat
from IPython.utils.terminal import get_terminal_size
def format_exception_as_a_whole(self, etype: type, evalue: Optional[BaseException], etb: Optional[TracebackType], number_of_lines_of_context, tb_offset: Optional[int]):
    """Formats the header, traceback and exception message for a single exception.

        This may be called multiple times by Python 3 exception chaining
        (PEP 3134).
        """
    orig_etype = etype
    try:
        etype = etype.__name__
    except AttributeError:
        pass
    tb_offset = self.tb_offset if tb_offset is None else tb_offset
    assert isinstance(tb_offset, int)
    head = self.prepare_header(str(etype), self.long_header)
    records = self.get_records(etb, number_of_lines_of_context, tb_offset) if etb else []
    frames = []
    skipped = 0
    lastrecord = len(records) - 1
    for i, record in enumerate(records):
        if not isinstance(record._sd, stack_data.RepeatedFrames) and self.skip_hidden:
            if record.frame.f_locals.get('__tracebackhide__', 0) and i != lastrecord:
                skipped += 1
                continue
        if skipped:
            Colors = self.Colors
            ColorsNormal = Colors.Normal
            frames.append('    %s[... skipping hidden %s frame]%s\n' % (Colors.excName, skipped, ColorsNormal))
            skipped = 0
        frames.append(self.format_record(record))
    if skipped:
        Colors = self.Colors
        ColorsNormal = Colors.Normal
        frames.append('    %s[... skipping hidden %s frame]%s\n' % (Colors.excName, skipped, ColorsNormal))
    formatted_exception = self.format_exception(etype, evalue)
    if records:
        frame_info = records[-1]
        ipinst = get_ipython()
        if ipinst is not None:
            ipinst.hooks.synchronize_with_editor(frame_info.filename, frame_info.lineno, 0)
    return [[head] + frames + formatted_exception]