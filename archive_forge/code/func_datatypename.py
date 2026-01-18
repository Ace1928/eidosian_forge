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
def datatypename(datatype):
    if isinstance(datatype, wsme.types.UserType):
        return datatype.name
    if isinstance(datatype, wsme.types.DictType):
        return 'dict(%s: %s)' % (datatypename(datatype.key_type), datatypename(datatype.value_type))
    if isinstance(datatype, wsme.types.ArrayType):
        return 'list(%s)' % datatypename(datatype.item_type)
    return datatype.__name__