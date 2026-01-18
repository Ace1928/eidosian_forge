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
def render_dot_texinfo(self: TexinfoTranslator, node: graphviz, code: str, options: Dict, prefix: str='graphviz') -> None:
    try:
        fname, outfn = render_dot(self, code, options, 'png', prefix)
    except GraphvizError as exc:
        logger.warning(__('dot code %r: %s'), code, exc)
        raise nodes.SkipNode from exc
    if fname is not None:
        self.body.append('@image{%s,,,[graphviz],png}\n' % fname[:-4])
    raise nodes.SkipNode