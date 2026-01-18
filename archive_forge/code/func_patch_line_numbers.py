from __future__ import annotations
from functools import lru_cache
from typing import Any
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst.states import Text
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options
from sphinx.ext.napoleon.docstring import GoogleDocstring
from .attributes_patch import patch_attribute_handling
def patch_line_numbers() -> None:
    """Make the rst parser put line numbers on more nodes.

    When the line numbers are missing, we have a hard time placing the :rtype:.
    """
    Text.indent = patched_text_indent
    BaseAdmonition.run = patched_base_admonition_run