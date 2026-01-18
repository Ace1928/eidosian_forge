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
def set_output_format(self, format: str) -> Markdown:
    """
        Set the output format for the class instance.

        Arguments:
            format: Must be a known value in `Markdown.output_formats`.

        """
    self.output_format = format.lower().rstrip('145')
    try:
        self.serializer = self.output_formats[self.output_format]
    except KeyError as e:
        valid_formats = list(self.output_formats.keys())
        valid_formats.sort()
        message = 'Invalid Output Format: "%s". Use one of %s.' % (self.output_format, '"' + '", "'.join(valid_formats) + '"')
        e.args = (message,) + e.args[1:]
        raise
    return self