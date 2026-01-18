from __future__ import annotations
import itertools
import shutil
import os
import textwrap
import typing as T
import collections
from . import build
from . import coredata
from . import environment
from . import mesonlib
from . import mintro
from . import mlog
from .ast import AstIDGenerator, IntrospectionInterpreter
from .mesonlib import MachineChoice, OptionKey
def print_aligned(self) -> None:
    """Do the actual printing.

        This prints the generated output in an aligned, pretty form. it aims
        for a total width of 160 characters, but will use whatever the tty
        reports it's value to be. Though this is much wider than the standard
        80 characters of terminals, and even than the newer 120, compressing
        it to those lengths makes the output hard to read.

        Each column will have a specific width, and will be line wrapped.
        """
    total_width = shutil.get_terminal_size(fallback=(160, 0))[0]
    _col = max(total_width // 5, 20)
    last_column = total_width - 3 * _col - 3
    four_column = (_col, _col, _col, last_column if last_column > 1 else _col)
    for line in zip(self.name_col, self.value_col, self.choices_col, self.descr_col):
        if not any(line):
            mlog.log('')
            continue
        if line[0] and (not any(line[1:])):
            mlog.log(line[0])
            continue

        def wrap_text(text: LOGLINE, width: int) -> mlog.TV_LoggableList:
            raw = text.text if isinstance(text, mlog.AnsiDecorator) else text
            indent = ' ' if raw.startswith('[') else ''
            wrapped_ = textwrap.wrap(raw, width, subsequent_indent=indent)
            if isinstance(text, mlog.AnsiDecorator):
                wrapped = T.cast('T.List[LOGLINE]', [mlog.AnsiDecorator(i, text.code) for i in wrapped_])
            else:
                wrapped = T.cast('T.List[LOGLINE]', wrapped_)
            return [str(i) + ' ' * (width - len(i)) for i in wrapped]
        name = wrap_text(line[0], four_column[0])
        val = wrap_text(line[1], four_column[1])
        choice = wrap_text(line[2], four_column[2])
        desc = wrap_text(line[3], four_column[3])
        for l in itertools.zip_longest(name, val, choice, desc, fillvalue=''):
            items = [l[i] if l[i] else ' ' * four_column[i] for i in range(4)]
            mlog.log(*items)