import io
import logging
import os
import pkgutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import Logger
from typing import IO, Any, Iterable, Iterator, List, Optional, Tuple, Union, cast
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.tokenize import GoodTokenInfo
from blib2to3.pytree import NL
from . import grammar, parse, pgen, token, tokenize
def can_advance(self, to: int) -> bool:
    try:
        self.eat(to)
    except StopIteration:
        return False
    else:
        return True