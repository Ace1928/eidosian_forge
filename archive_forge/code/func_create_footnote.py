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
def create_footnote(self, uri: str, docname: str) -> Tuple[nodes.footnote, nodes.footnote_reference]:
    reference = nodes.reference('', nodes.Text(uri), refuri=uri, nolinkurl=True)
    footnote = nodes.footnote(uri, auto=1, docname=docname)
    footnote['names'].append('#')
    footnote += nodes.label('', '#')
    footnote += nodes.paragraph('', '', reference)
    self.document.note_autofootnote(footnote)
    footnote_ref = nodes.footnote_reference('[#]_', auto=1, refid=footnote['ids'][0], docname=docname)
    footnote_ref += nodes.Text('#')
    self.document.note_autofootnote_ref(footnote_ref)
    footnote.add_backref(footnote_ref['ids'][0])
    return (footnote, footnote_ref)