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
def patched_parse_annotation(settings: Values, typ: str, env: Any) -> Any:
    if not typ.startswith(TYPE_IS_RST_LABEL):
        return _parse_annotation(typ, env)
    typ = typ[len(TYPE_IS_RST_LABEL):]
    return rst_to_docutils(settings, typ)