import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
def help_pager(out_stream: Optional[TextIO]=None) -> autopage.AutoPager:
    """Return an AutoPager suitable for help output."""
    return autopage.AutoPager(out_stream, allow_color=True, line_buffering=False, reset_on_exit=False)