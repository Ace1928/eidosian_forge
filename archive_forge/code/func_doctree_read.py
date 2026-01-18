from typing import Any, Dict, Set
from docutils import nodes
from docutils.nodes import Node
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.locale import _
def doctree_read(app: Sphinx, doctree: Node) -> None:
    env = app.builder.env
    resolve_target = getattr(env.config, 'linkcode_resolve', None)
    if not callable(env.config.linkcode_resolve):
        raise LinkcodeError('Function `linkcode_resolve` is not given in conf.py')
    domain_keys = {'py': ['module', 'fullname'], 'c': ['names'], 'cpp': ['names'], 'js': ['object', 'fullname']}
    for objnode in list(doctree.findall(addnodes.desc)):
        domain = objnode.get('domain')
        uris: Set[str] = set()
        for signode in objnode:
            if not isinstance(signode, addnodes.desc_signature):
                continue
            info = {}
            for key in domain_keys.get(domain, []):
                value = signode.get(key)
                if not value:
                    value = ''
                info[key] = value
            if not info:
                continue
            uri = resolve_target(domain, info)
            if not uri:
                continue
            if uri in uris or not uri:
                continue
            uris.add(uri)
            inline = nodes.inline('', _('[source]'), classes=['viewcode-link'])
            onlynode = addnodes.only(expr='html')
            onlynode += nodes.reference('', '', inline, internal=False, refuri=uri)
            signode += onlynode