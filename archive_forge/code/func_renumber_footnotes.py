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
def renumber_footnotes(self) -> None:
    collector = FootnoteCollector(self.document)
    self.document.walkabout(collector)
    num = 0
    for footnote in collector.auto_footnotes:
        while True:
            num += 1
            if str(num) not in collector.used_footnote_numbers:
                break
        old_label = cast(nodes.label, footnote[0])
        old_label.replace_self(nodes.label('', str(num)))
        if old_label in footnote['names']:
            footnote['names'].remove(old_label.astext())
        footnote['names'].append(str(num))
        docname = footnote['docname']
        for ref in collector.footnote_refs:
            if docname == ref['docname'] and footnote['ids'][0] == ref['refid']:
                ref.remove(ref[0])
                ref += nodes.Text(str(num))