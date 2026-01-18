from __future__ import annotations
import http
import logging
import sys
from copy import copy
from typing import Literal
import click
def should_use_colors(self) -> bool:
    return sys.stderr.isatty()