from __future__ import annotations
from . import Extension
from ..blockprocessors import OListProcessor, UListProcessor
import re
from typing import TYPE_CHECKING
def extendMarkdown(self, md):
    """ Override existing Processors. """
    md.parser.blockprocessors.register(SaneOListProcessor(md.parser), 'olist', 40)
    md.parser.blockprocessors.register(SaneUListProcessor(md.parser), 'ulist', 30)