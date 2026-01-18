import re
import textwrap
from os import path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Pattern, Set,
from docutils import nodes, writers
from docutils.nodes import Element, Node, Text
from sphinx import __display_version__, addnodes
from sphinx.domains import IndexEntry
from sphinx.domains.index import IndexDomain
from sphinx.errors import ExtensionError
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.i18n import format_date
from sphinx.writers.latex import collected_footnote
def collect_node_menus(self) -> None:
    """Collect the menu entries for each "node" section."""
    node_menus = self.node_menus
    targets: List[Element] = [self.document]
    targets.extend(self.document.findall(nodes.section))
    for node in targets:
        assert 'node_name' in node and node['node_name']
        entries = [s['node_name'] for s in find_subsections(node)]
        node_menus[node['node_name']] = entries
    title = self.document.next_node(nodes.title)
    top = title.parent if title else self.document
    if not isinstance(top, (nodes.document, nodes.section)):
        top = self.document
    if top is not self.document:
        entries = node_menus[top['node_name']]
        entries += node_menus['Top'][1:]
        node_menus['Top'] = entries
        del node_menus[top['node_name']]
        top['node_name'] = 'Top'
    for name, _content in self.indices:
        node_menus[name] = []
        node_menus['Top'].append(name)