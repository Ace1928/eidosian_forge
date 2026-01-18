from __future__ import annotations
import codecs
import sys
import logging
import importlib
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, ClassVar, Mapping, Sequence
from . import util
from .preprocessors import build_preprocessors
from .blockprocessors import build_block_parser
from .treeprocessors import build_treeprocessors
from .inlinepatterns import build_inlinepatterns
from .postprocessors import build_postprocessors
from .extensions import Extension
from .serializers import to_html_string, to_xhtml_string
from .util import BLOCK_LEVEL_ELEMENTS
def is_block_level(self, tag: Any) -> bool:
    """
        Check if the given `tag` is a block level HTML tag.

        Returns `True` for any string listed in `Markdown.block_level_elements`. A `tag` which is
        not a string always returns `False`.

        """
    if isinstance(tag, str):
        return tag.lower().rstrip('/') in self.block_level_elements
    return False