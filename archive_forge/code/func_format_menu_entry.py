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
def format_menu_entry(self, name: str, node_name: str, desc: str) -> str:
    if name == node_name:
        s = '* %s:: ' % (name,)
    else:
        s = '* %s: %s. ' % (name, node_name)
    offset = max((24, (len(name) + 4) % 78))
    wdesc = '\n'.join((' ' * offset + l for l in textwrap.wrap(desc, width=78 - offset)))
    return s + wdesc.strip() + '\n'