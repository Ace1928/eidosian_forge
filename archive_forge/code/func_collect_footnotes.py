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
def collect_footnotes(self, node: Element) -> Dict[str, List[Union[collected_footnote, bool]]]:

    def footnotes_under(n: Element) -> Iterator[nodes.footnote]:
        if isinstance(n, nodes.footnote):
            yield n
        else:
            for c in n.children:
                if isinstance(c, addnodes.start_of_file):
                    continue
                elif isinstance(c, nodes.Element):
                    yield from footnotes_under(c)
    fnotes: Dict[str, List[Union[collected_footnote, bool]]] = {}
    for fn in footnotes_under(node):
        label = cast(nodes.label, fn[0])
        num = label.astext().strip()
        fnotes[num] = [collected_footnote('', *fn.children), False]
    return fnotes