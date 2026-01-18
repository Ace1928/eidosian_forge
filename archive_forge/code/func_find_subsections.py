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
def find_subsections(section: Element) -> List[nodes.section]:
    """Return a list of subsections for the given ``section``."""
    result = []
    for child in section:
        if isinstance(child, nodes.section):
            result.append(child)
            continue
        elif isinstance(child, nodes.Element):
            result.extend(find_subsections(child))
    return result