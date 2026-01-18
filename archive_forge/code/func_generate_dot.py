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
def generate_dot(self, name: str, urls: Dict[str, str]={}, env: Optional[BuildEnvironment]=None, graph_attrs: Dict={}, node_attrs: Dict={}, edge_attrs: Dict={}) -> str:
    """Generate a graphviz dot graph from the classes that were passed in
        to __init__.

        *name* is the name of the graph.

        *urls* is a dictionary mapping class names to HTTP URLs.

        *graph_attrs*, *node_attrs*, *edge_attrs* are dictionaries containing
        key/value pairs to pass on as graphviz properties.
        """
    g_attrs = self.default_graph_attrs.copy()
    n_attrs = self.default_node_attrs.copy()
    e_attrs = self.default_edge_attrs.copy()
    g_attrs.update(graph_attrs)
    n_attrs.update(node_attrs)
    e_attrs.update(edge_attrs)
    if env:
        g_attrs.update(env.config.inheritance_graph_attrs)
        n_attrs.update(env.config.inheritance_node_attrs)
        e_attrs.update(env.config.inheritance_edge_attrs)
    res: List[str] = []
    res.append('digraph %s {\n' % name)
    res.append(self._format_graph_attrs(g_attrs))
    for name, fullname, bases, tooltip in sorted(self.class_info):
        this_node_attrs = n_attrs.copy()
        if fullname in urls:
            this_node_attrs['URL'] = '"%s"' % urls[fullname]
            this_node_attrs['target'] = '"_top"'
        if tooltip:
            this_node_attrs['tooltip'] = tooltip
        res.append('  "%s" [%s];\n' % (name, self._format_node_attrs(this_node_attrs)))
        for base_name in bases:
            res.append('  "%s" -> "%s" [%s];\n' % (base_name, name, self._format_node_attrs(e_attrs)))
    res.append('}\n')
    return ''.join(res)