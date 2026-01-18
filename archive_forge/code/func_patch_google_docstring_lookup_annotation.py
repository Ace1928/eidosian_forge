from __future__ import annotations
from functools import lru_cache
from typing import Any
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst.states import Text
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options
from sphinx.ext.napoleon.docstring import GoogleDocstring
from .attributes_patch import patch_attribute_handling
def patch_google_docstring_lookup_annotation() -> None:
    """Fix issue 308:
    https://github.com/tox-dev/sphinx-autodoc-typehints/issues/308
    """
    GoogleDocstring._lookup_annotation = patched_lookup_annotation