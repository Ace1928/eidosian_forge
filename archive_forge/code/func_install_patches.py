from __future__ import annotations
from functools import lru_cache
from typing import Any
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst.states import Text
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options
from sphinx.ext.napoleon.docstring import GoogleDocstring
from .attributes_patch import patch_attribute_handling
def install_patches(app: Sphinx) -> None:
    fix_autodoc_typehints_for_overloaded_methods()
    patch_attribute_handling(app)
    patch_google_docstring_lookup_annotation()
    fix_napoleon_numpy_docstring_return_type(app)
    patch_line_numbers()