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
def escape_arg(self, s: str) -> str:
    """Return an escaped string suitable for use as an argument
        to a Texinfo command."""
    s = self.escape(s)
    s = s.replace(',', '@comma{}')
    s = ' '.join(s.split()).strip()
    return s