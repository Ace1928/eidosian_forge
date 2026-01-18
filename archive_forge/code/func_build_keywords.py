from __future__ import annotations
import html
import os
import posixpath
import re
from collections.abc import Iterable
from os import path
from typing import Any, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import canon_path, make_filename
from sphinx.util.template import SphinxRenderer
def build_keywords(self, title: str, refs: list[Any], subitems: Any) -> list[str]:
    keywords: list[str] = []
    if len(refs) == 1:
        keywords.append(self.keyword_item(title, refs[0]))
    elif len(refs) > 1:
        for i, ref in enumerate(refs):
            keywords.append(self.keyword_item(title, ref))
    if subitems:
        for subitem in subitems:
            keywords.extend(self.build_keywords(subitem[0], subitem[1], []))
    return keywords