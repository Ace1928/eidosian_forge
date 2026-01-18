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
def markdownFromFile(**kwargs: Any):
    """
    Read Markdown text from a file and write output to a file or a stream.

    This is a shortcut function which initializes an instance of [`Markdown`][markdown.Markdown],
    and calls the [`convertFile`][markdown.Markdown.convertFile] method rather than
    [`convert`][markdown.Markdown.convert].

    Keyword arguments:
        input (str | BinaryIO): A file name or readable object.
        output (str | BinaryIO): A file name or writable object.
        encoding (str): Encoding of input and output.
        **kwargs: Any arguments accepted by the `Markdown` class.

    """
    md = Markdown(**kwargs)
    md.convertFile(kwargs.get('input', None), kwargs.get('output', None), kwargs.get('encoding', None))