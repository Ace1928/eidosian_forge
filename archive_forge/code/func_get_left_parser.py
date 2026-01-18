from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
@abc.abstractmethod
def get_left_parser(self, state: ParserState) -> Parser:
    """Return the parser for the left side."""