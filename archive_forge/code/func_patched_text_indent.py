from __future__ import annotations
from functools import lru_cache
from typing import Any
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst.states import Text
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options
from sphinx.ext.napoleon.docstring import GoogleDocstring
from .attributes_patch import patch_attribute_handling
def patched_text_indent(self: Text, *args: Any) -> Any:
    _, line = self.state_machine.get_source_and_line()
    result = orig_text_indent(self, *args)
    node = self.parent[-1]
    if node.tagname == 'system_message':
        node = self.parent[-2]
    node.line = line
    return result