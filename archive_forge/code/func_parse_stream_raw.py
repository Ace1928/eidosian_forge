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
def parse_stream_raw(self, stream: IO[str], debug: bool=False) -> NL:
    """Parse a stream and return the syntax tree."""
    tokens = tokenize.generate_tokens(stream.readline, grammar=self.grammar)
    return self.parse_tokens(tokens, debug)