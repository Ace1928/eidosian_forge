from __future__ import annotations
from functools import lru_cache
from typing import Any
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst.states import Text
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options
from sphinx.ext.napoleon.docstring import GoogleDocstring
from .attributes_patch import patch_attribute_handling
@lru_cache()
def fix_autodoc_typehints_for_overloaded_methods() -> None:
    """
    sphinx-autodoc-typehints responds to the "autodoc-process-signature" event
    to remove types from the signature line of functions.

    Normally, `FunctionDocumenter.format_signature` and
    `MethodDocumenter.format_signature` call `super().format_signature` which
    ends up going to `Documenter.format_signature`, and this last method emits
    the `autodoc-process-signature` event. However, if there are overloads,
    `FunctionDocumenter.format_signature` does something else and the event
    never occurs.

    Here we remove this alternative code path by brute force.

    See https://github.com/tox-dev/sphinx-autodoc-typehints/issues/296
    """
    from sphinx.ext.autodoc import FunctionDocumenter, MethodDocumenter
    del FunctionDocumenter.format_signature
    del MethodDocumenter.format_signature