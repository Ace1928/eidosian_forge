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
def get_fignumber(self, env: 'BuildEnvironment', builder: 'Builder', figtype: str, docname: str, target_node: Element) -> Tuple[int, ...]:
    if figtype == 'section':
        if builder.name == 'latex':
            return ()
        elif docname not in env.toc_secnumbers:
            raise ValueError
        else:
            anchorname = '#' + target_node['ids'][0]
            if anchorname not in env.toc_secnumbers[docname]:
                return env.toc_secnumbers[docname].get('')
            else:
                return env.toc_secnumbers[docname].get(anchorname)
    else:
        try:
            figure_id = target_node['ids'][0]
            return env.toc_fignumbers[docname][figtype][figure_id]
        except (KeyError, IndexError) as exc:
            raise ValueError from exc