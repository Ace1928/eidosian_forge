import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
def color_metavar(size: int) -> Tuple[str, ...]:
    return tuple((f'\x1b[3m{mv}\x1b[0m' for mv in get_metavars(size)))