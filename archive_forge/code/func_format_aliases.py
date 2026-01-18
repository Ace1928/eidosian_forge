from __future__ import annotations
import typing as t
from collections import defaultdict
from textwrap import dedent
from traitlets import HasTraits, Undefined
from traitlets.config.application import Application
from traitlets.utils.text import indent
def format_aliases(aliases: list[str]) -> str:
    fmted = []
    for a in aliases:
        dashes = '-' if len(a) == 1 else '--'
        fmted.append(f'``{dashes}{a}``')
    return ', '.join(fmted)