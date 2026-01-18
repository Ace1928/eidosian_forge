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
def convertFile(self, input: str | BinaryIO | None=None, output: str | BinaryIO | None=None, encoding: str | None=None) -> Markdown:
    """
        Converts a Markdown file and returns the HTML as a Unicode string.

        Decodes the file using the provided encoding (defaults to `utf-8`),
        passes the file content to markdown, and outputs the HTML to either
        the provided stream or the file with provided name, using the same
        encoding as the source file. The
        [`xmlcharrefreplace`](https://docs.python.org/3/library/codecs.html#error-handlers)
        error handler is used when encoding the output.

        **Note:** This is the only place that decoding and encoding of Unicode
        takes place in Python-Markdown.  (All other code is Unicode-in /
        Unicode-out.)

        Arguments:
            input: File object or path. Reads from `stdin` if `None`.
            output: File object or path. Writes to `stdout` if `None`.
            encoding: Encoding of input and output files. Defaults to `utf-8`.

        """
    encoding = encoding or 'utf-8'
    if input:
        if isinstance(input, str):
            input_file = codecs.open(input, mode='r', encoding=encoding)
        else:
            input_file = codecs.getreader(encoding)(input)
        text = input_file.read()
        input_file.close()
    else:
        text = sys.stdin.read()
    text = text.lstrip('\ufeff')
    html = self.convert(text)
    if output:
        if isinstance(output, str):
            output_file = codecs.open(output, 'w', encoding=encoding, errors='xmlcharrefreplace')
            output_file.write(html)
            output_file.close()
        else:
            writer = codecs.getwriter(encoding)
            output_file = writer(output, errors='xmlcharrefreplace')
            output_file.write(html)
    else:
        html = html.encode(encoding, 'xmlcharrefreplace')
        sys.stdout.buffer.write(html)
    return self