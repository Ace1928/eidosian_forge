import re
import unicodedata
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple, Type,
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging
def extract_messages(doctree: Element) -> Iterable[Tuple[Element, str]]:
    """Extract translatable messages from a document tree."""
    for node in doctree.findall(is_translatable):
        if isinstance(node, addnodes.translatable):
            for msg in node.extract_original_messages():
                yield (node, msg)
            continue
        if isinstance(node, LITERAL_TYPE_NODES):
            msg = node.rawsource
            if not msg:
                msg = node.astext()
        elif isinstance(node, nodes.image):
            if node.get('alt'):
                yield (node, node['alt'])
            if node.get('translatable'):
                msg = '.. image:: %s' % node['uri']
            else:
                msg = ''
        elif isinstance(node, META_TYPE_NODES):
            msg = node.rawcontent
        elif isinstance(node, nodes.pending) and is_pending_meta(node):
            msg = node.details['nodes'][0].rawcontent
        elif isinstance(node, addnodes.docutils_meta):
            msg = node['content']
        else:
            msg = node.rawsource.replace('\n', ' ').strip()
        if msg:
            yield (node, msg)