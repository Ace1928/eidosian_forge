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
def scan_services(service, path=[]):
    has_functions = False
    for name in dir(service):
        if name.startswith('_'):
            continue
        a = getattr(service, name)
        if inspect.ismethod(a):
            if hasattr(a, '_wsme_definition'):
                has_functions = True
        if inspect.isclass(a):
            continue
        if len(path) > wsme.rest.APIPATH_MAXLEN:
            raise ValueError('Path is too long: ' + str(path))
        for value in scan_services(a, path + [name]):
            yield value
    if has_functions:
        yield (service, path)