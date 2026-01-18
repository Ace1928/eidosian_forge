import doctest
import re
import sys
import time
from io import StringIO
from os import path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement
from docutils.parsers.rst import directives
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version
import sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relpath
from sphinx.util.typing import OptionSpec
def skipped(self, node: Element) -> bool:
    if 'skipif' not in node:
        return False
    else:
        condition = node['skipif']
        context: Dict[str, Any] = {}
        if self.config.doctest_global_setup:
            exec(self.config.doctest_global_setup, context)
        should_skip = eval(condition, context)
        if self.config.doctest_global_cleanup:
            exec(self.config.doctest_global_cleanup, context)
        return should_skip