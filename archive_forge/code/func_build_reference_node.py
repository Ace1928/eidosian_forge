import re
import sys
from copy import copy
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional,
from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.locale import _, __
from sphinx.roles import EmphasizedLiteral, XRefRole
from sphinx.util import docname_join, logging, ws_re
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import clean_astext, make_id, make_refnode
from sphinx.util.typing import OptionSpec, RoleFunction
def build_reference_node(self, fromdocname: str, builder: 'Builder', docname: str, labelid: str, sectname: str, rolename: str, **options: Any) -> Element:
    nodeclass = options.pop('nodeclass', nodes.reference)
    newnode = nodeclass('', '', internal=True, **options)
    innernode = nodes.inline(sectname, sectname)
    if innernode.get('classes') is not None:
        innernode['classes'].append('std')
        innernode['classes'].append('std-' + rolename)
    if docname == fromdocname:
        newnode['refid'] = labelid
    else:
        contnode = pending_xref('')
        contnode['refdocname'] = docname
        contnode['refsectname'] = sectname
        newnode['refuri'] = builder.get_relative_uri(fromdocname, docname)
        if labelid:
            newnode['refuri'] += '#' + labelid
    newnode.append(innernode)
    return newnode