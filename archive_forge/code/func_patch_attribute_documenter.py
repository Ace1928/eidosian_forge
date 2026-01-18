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
def patch_attribute_documenter(app: Sphinx) -> None:
    """Instead of using stringify_typehint in
    `AttributeDocumenter.add_directive_header`, use `format_annotation`
    """

    def add_directive_header(*args: Any, **kwargs: Any) -> Any:
        with patch(STRINGIFY_PATCH_TARGET, partial(stringify_annotation, app)):
            return orig_add_directive_header(*args, **kwargs)
    AttributeDocumenter.add_directive_header = add_directive_header