from typing import Any, Dict, List, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.transforms.references import Substitutions
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.latex.nodes import (captioned_literal_block, footnotemark, footnotetext,
from sphinx.domains.citation import CitationDomain
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.nodes import NodeMatcher
def expand_show_urls(self) -> None:
    show_urls = self.config.latex_show_urls
    if show_urls is False or show_urls == 'no':
        return
    for node in list(self.document.findall(nodes.reference)):
        uri = node.get('refuri', '')
        if uri.startswith(URI_SCHEMES):
            if uri.startswith('mailto:'):
                uri = uri[7:]
            if node.astext() != uri:
                index = node.parent.index(node)
                docname = self.get_docname_for_node(node)
                if show_urls == 'footnote':
                    fn, fnref = self.create_footnote(uri, docname)
                    node.parent.insert(index + 1, fn)
                    node.parent.insert(index + 2, fnref)
                    self.expanded = True
                else:
                    textnode = nodes.Text(' (%s)' % uri)
                    node.parent.insert(index + 1, textnode)