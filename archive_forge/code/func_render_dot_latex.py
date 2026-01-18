import posixpath
import re
import subprocess
from os import path
from subprocess import PIPE, CalledProcessError
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging, sha1
from sphinx.util.docutils import SphinxDirective, SphinxTranslator
from sphinx.util.fileutil import copy_asset
from sphinx.util.i18n import search_image_for_language
from sphinx.util.nodes import set_source_info
from sphinx.util.osutil import ensuredir
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
from sphinx.writers.manpage import ManualPageTranslator
from sphinx.writers.texinfo import TexinfoTranslator
from sphinx.writers.text import TextTranslator
def render_dot_latex(self: LaTeXTranslator, node: graphviz, code: str, options: Dict, prefix: str='graphviz', filename: Optional[str]=None) -> None:
    try:
        fname, outfn = render_dot(self, code, options, 'pdf', prefix, filename)
    except GraphvizError as exc:
        logger.warning(__('dot code %r: %s'), code, exc)
        raise nodes.SkipNode from exc
    is_inline = self.is_inline(node)
    if not is_inline:
        pre = ''
        post = ''
        if 'align' in node:
            if node['align'] == 'left':
                pre = '{'
                post = '\\hspace*{\\fill}}'
            elif node['align'] == 'right':
                pre = '{\\hspace*{\\fill}'
                post = '}'
            elif node['align'] == 'center':
                pre = '{\\hfill'
                post = '\\hspace*{\\fill}}'
        self.body.append('\n%s' % pre)
    self.body.append('\\sphinxincludegraphics[]{%s}' % fname)
    if not is_inline:
        self.body.append('%s\n' % post)
    raise nodes.SkipNode