from __future__ import annotations
from functools import lru_cache
from typing import Any
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst.states import Text
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options
from sphinx.ext.napoleon.docstring import GoogleDocstring
from .attributes_patch import patch_attribute_handling
def fix_napoleon_numpy_docstring_return_type(app: Sphinx) -> None:
    """
    If no return type is explicitly provided, numpy docstrings will mess up and
    use the return type text as return types.
    """
    app.connect('autodoc-process-docstring', napoleon_numpy_docstring_return_type_processor, priority=499)