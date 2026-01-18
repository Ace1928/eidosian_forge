from __future__ import annotations
import logging  # isort:skip
from os.path import basename, join
from docutils import nodes
from docutils.parsers.rst.directives import unchanged
from sphinx.directives.code import CodeBlock, container_wrapper, dedent_lines
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import logging, parselinenos
from sphinx.util.nodes import set_source_info
from . import PARALLEL_SAFE
from .templates import (
from .util import get_sphinx_resources
def get_codeblock_node(self, code, language):
    """this is copied from sphinx.directives.code.CodeBlock.run

        it has been changed to accept code and language as an arguments instead
        of reading from self

        """
    document = self.state.document
    location = self.state_machine.get_source_and_line(self.lineno)
    linespec = self.options.get('emphasize-lines')
    if linespec:
        try:
            nlines = len(code.split('\n'))
            hl_lines = parselinenos(linespec, nlines)
            if any((i >= nlines for i in hl_lines)):
                emph_lines = self.options['emphasize-lines']
                log.warning(__(f'line number spec is out of range(1-{nlines}): {emph_lines!r}'), location=location)
            hl_lines = [x + 1 for x in hl_lines if x < nlines]
        except ValueError as err:
            return [document.reporter.warning(str(err), line=self.lineno)]
    else:
        hl_lines = None
    if 'dedent' in self.options:
        location = self.state_machine.get_source_and_line(self.lineno)
        lines = code.split('\n')
        lines = dedent_lines(lines, self.options['dedent'], location=location)
        code = '\n'.join(lines)
    literal = nodes.literal_block(code, code)
    literal['language'] = language
    literal['linenos'] = 'linenos' in self.options or 'lineno-start' in self.options
    literal['classes'] += self.options.get('class', [])
    extra_args = literal['highlight_args'] = {}
    if hl_lines is not None:
        extra_args['hl_lines'] = hl_lines
    if 'lineno-start' in self.options:
        extra_args['linenostart'] = self.options['lineno-start']
    set_source_info(self, literal)
    caption = self.options.get('caption')
    if caption:
        try:
            literal = container_wrapper(self, literal, caption)
        except ValueError as exc:
            return [document.reporter.warning(str(exc), line=self.lineno)]
    self.add_name(literal)
    return [literal]