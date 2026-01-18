from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.typing import TextlikeNode
def make_xref(self, rolename: str, domain: str, target: str, innernode: Type[TextlikeNode]=addnodes.literal_emphasis, contnode: Node=None, env: BuildEnvironment=None, inliner: Inliner=None, location: Node=None) -> Node:
    assert env is not None
    assert (inliner is None) == (location is None), (inliner, location)
    if not rolename:
        return contnode or innernode(target, target)
    role = env.get_domain(domain).role(rolename)
    if role is None or inliner is None:
        if role is None and inliner is not None:
            msg = __("Problem in %s domain: field is supposed to use role '%s', but that role is not in the domain.")
            logger.warning(__(msg), domain, rolename, location=location)
        refnode = addnodes.pending_xref('', refdomain=domain, refexplicit=False, reftype=rolename, reftarget=target)
        refnode += contnode or innernode(target, target)
        env.get_domain(domain).process_field_xref(refnode)
        return refnode
    lineno = logging.get_source_line(location)[1]
    ns, messages = role(rolename, target, target, lineno, inliner, {}, [])
    return nodes.inline(target, '', *ns)