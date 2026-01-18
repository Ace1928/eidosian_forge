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
def document_function(funcdef, docstrings=None, protocols=['restjson']):
    """A helper function to complete a function documentation with return and
    parameter types"""
    if not docstrings:
        docstrings = [[]]
    found_params = set()
    for si, docstring in enumerate(docstrings):
        for i, line in enumerate(docstring):
            m = field_re.match(line)
            if m and m.group('field') == 'param':
                found_params.add(m.group('name'))
    next_param_pos = (0, 0)
    for arg in funcdef.arguments:
        content = [u':type  %s: :wsme:type:`%s`' % (arg.name, datatypename(arg.datatype))]
        if arg.name not in found_params:
            content.insert(0, u':param %s: ' % arg.name)
            pos = next_param_pos
        else:
            for si, docstring in enumerate(docstrings):
                for i, line in enumerate(docstring):
                    m = field_re.match(line)
                    if m and m.group('field') == 'param' and (m.group('name') == arg.name):
                        pos = (si, i + 1)
                        break
        docstring = docstrings[pos[0]]
        docstring[pos[1]:pos[1]] = content
        next_param_pos = (pos[0], pos[1] + len(content))
    if funcdef.return_type:
        content = [u':rtype: %s' % datatypename(funcdef.return_type)]
        pos = None
        for si, docstring in enumerate(docstrings):
            for i, line in enumerate(docstring):
                m = field_re.match(line)
                if m and m.group('field') == 'return':
                    pos = (si, i + 1)
                    break
        else:
            pos = next_param_pos
        docstring = docstrings[pos[0]]
        docstring[pos[1]:pos[1]] = content
    codesamples = []
    if protocols:
        params = []
        for arg in funcdef.arguments:
            params.append((arg.name, arg.datatype, make_sample_object(arg.datatype)))
        codesamples.extend([u':%s:' % _(u'Parameters samples'), u'    .. cssclass:: toggle', u''])
        for name, protocol in protocols:
            language, sample = protocol.encode_sample_params(params, format=True)
            codesamples.extend([u' ' * 4 + name, u'        .. code-block:: ' + language, u''])
            codesamples.extend((u' ' * 12 + line for line in str(sample).split('\n')))
        if funcdef.return_type:
            codesamples.extend([u':%s:' % _(u'Return samples'), u'    .. cssclass:: toggle', u''])
            sample_obj = make_sample_object(funcdef.return_type)
            for name, protocol in protocols:
                language, sample = protocol.encode_sample_result(funcdef.return_type, sample_obj, format=True)
                codesamples.extend([u' ' * 4 + name, u'        .. code-block:: ' + language, u''])
                codesamples.extend((u' ' * 12 + line for line in str(sample).split('\n')))
    docstrings[0:0] = [codesamples]
    return docstrings