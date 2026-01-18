import inspect
import re
import sys
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain
from sphinx.domains import ObjType
from sphinx.domains.python import PyAttribute
from sphinx.domains.python import PyClasslike
from sphinx.domains.python import PyMethod
from sphinx.ext import autodoc
from sphinx.locale import _
from sphinx.roles import XRefRole
from sphinx.util.docfields import Field
from sphinx.util.nodes import make_refnode
import wsme
import wsme.rest.json
import wsme.rest.xml
import wsme.types
def check_samples_slot(value):
    """Validate the samples_slot option to the TypeDocumenter.

    Valid positions are 'before-docstring' and
    'after-docstring'. Using the explicit 'none' disables sample
    output. The default is after-docstring.
    """
    if not value:
        return 'after-docstring'
    val = directives.choice(value, ('none', 'before-docstring', 'after-docstring'))
    return val