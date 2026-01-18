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
def add_menu_entries(self, entries: List[str], reg: Pattern=re.compile('\\s+---?\\s+')) -> None:
    for entry in entries:
        name = self.node_names[entry]
        try:
            parts = reg.split(name, 1)
        except TypeError:
            parts = [name]
        if len(parts) == 2:
            name, desc = parts
        else:
            desc = ''
        name = self.escape_menu(name)
        desc = self.escape(desc)
        self.body.append(self.format_menu_entry(name, entry, desc))