from functools import partial
from optparse import Values
from typing import Any, Tuple
from unittest.mock import patch
import sphinx.domains.python
import sphinx.ext.autodoc
from docutils.parsers.rst import Parser as RstParser
from docutils.utils import new_document
from sphinx.addnodes import desc_signature
from sphinx.application import Sphinx
from sphinx.domains.python import PyAttribute
from sphinx.ext.autodoc import AttributeDocumenter
def stringify_annotation(app: Sphinx, annotation: Any, mode: str='') -> str:
    """Format the annotation with sphinx-autodoc-typehints and inject our
    magic prefix to tell our patched PyAttribute.handle_signature to treat
    it as rst."""
    from . import format_annotation
    return TYPE_IS_RST_LABEL + format_annotation(annotation, app.config)