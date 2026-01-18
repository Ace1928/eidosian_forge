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
def get_theme_config(self) -> tuple[str, dict]:
    return (self.config.qthelp_theme, self.config.qthelp_theme_options)