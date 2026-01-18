import builtins
import inspect
import re
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.graphviz import (figure_wrapper, graphviz, render_dot_html, render_dot_latex,
from sphinx.util import md5
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
from sphinx.writers.texinfo import TexinfoTranslator
def html_visit_inheritance_diagram(self: HTMLTranslator, node: inheritance_diagram) -> None:
    """
    Output the graph for HTML.  This will insert a PNG with clickable
    image map.
    """
    graph = node['graph']
    graph_hash = get_graph_hash(node)
    name = 'inheritance%s' % graph_hash
    graphviz_output_format = self.builder.env.config.graphviz_output_format.upper()
    current_filename = self.builder.current_docname + self.builder.out_suffix
    urls = {}
    pending_xrefs = cast(Iterable[addnodes.pending_xref], node)
    for child in pending_xrefs:
        if child.get('refuri') is not None:
            if graphviz_output_format == 'SVG':
                urls[child['reftitle']] = '../' + child.get('refuri')
            else:
                urls[child['reftitle']] = child.get('refuri')
        elif child.get('refid') is not None:
            if graphviz_output_format == 'SVG':
                urls[child['reftitle']] = '../' + current_filename + '#' + child.get('refid')
            else:
                urls[child['reftitle']] = '#' + child.get('refid')
    dotcode = graph.generate_dot(name, urls, env=self.builder.env)
    render_dot_html(self, node, dotcode, {}, 'inheritance', 'inheritance', alt='Inheritance diagram of ' + node['content'])
    raise nodes.SkipNode