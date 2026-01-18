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
def collect_indices(self) -> None:

    def generate(content: List[Tuple[str, List[IndexEntry]]], collapsed: bool) -> str:
        ret = ['\n@menu\n']
        for _letter, entries in content:
            for entry in entries:
                if not entry[3]:
                    continue
                name = self.escape_menu(entry[0])
                sid = self.get_short_id('%s:%s' % (entry[2], entry[3]))
                desc = self.escape_arg(entry[6])
                me = self.format_menu_entry(name, sid, desc)
                ret.append(me)
        ret.append('@end menu\n')
        return ''.join(ret)
    indices_config = self.config.texinfo_domain_indices
    if indices_config:
        for domain in self.builder.env.domains.values():
            for indexcls in domain.indices:
                indexname = '%s-%s' % (domain.name, indexcls.name)
                if isinstance(indices_config, list):
                    if indexname not in indices_config:
                        continue
                content, collapsed = indexcls(domain).generate(self.builder.docnames)
                if not content:
                    continue
                self.indices.append((indexcls.localname, generate(content, collapsed)))
    domain = cast(IndexDomain, self.builder.env.get_domain('index'))
    for docname in self.builder.docnames:
        if domain.entries[docname]:
            self.indices.append((_('Index'), '\n@printindex ge\n'))
            break