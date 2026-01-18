from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union, cast
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.environment.adapters.toctree import TocTree
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.locale import __
from sphinx.transforms import SphinxContentsFilter
from sphinx.util import logging, url_re
def build_toc(node: Union[Element, Sequence[Element]], depth: int=1) -> Optional[nodes.bullet_list]:
    entries: List[Element] = []
    memo_parents: Dict[Tuple[str, ...], nodes.list_item] = {}
    for sectionnode in node:
        if isinstance(sectionnode, nodes.section):
            title = sectionnode[0]
            visitor = SphinxContentsFilter(doctree)
            title.walkabout(visitor)
            nodetext = visitor.get_entry_text()
            anchorname = _make_anchor_name(sectionnode['ids'], numentries)
            reference = nodes.reference('', '', *nodetext, internal=True, refuri=docname, anchorname=anchorname)
            para = addnodes.compact_paragraph('', '', reference)
            item: Element = nodes.list_item('', para)
            sub_item = build_toc(sectionnode, depth + 1)
            if sub_item:
                item += sub_item
            entries.append(item)
        elif isinstance(sectionnode, addnodes.only):
            onlynode = addnodes.only(expr=sectionnode['expr'])
            blist = build_toc(sectionnode, depth)
            if blist:
                onlynode += blist.children
                entries.append(onlynode)
        elif isinstance(sectionnode, nodes.Element):
            toctreenode: nodes.Node
            for toctreenode in sectionnode.findall():
                if isinstance(toctreenode, nodes.section):
                    continue
                if isinstance(toctreenode, addnodes.toctree):
                    item = toctreenode.copy()
                    entries.append(item)
                    TocTree(app.env).note(docname, toctreenode)
                elif isinstance(toctreenode, addnodes.desc):
                    for sig_node in toctreenode:
                        if not isinstance(sig_node, addnodes.desc_signature):
                            continue
                        if not sig_node.get('_toc_name', ''):
                            continue
                        if sig_node.parent.get('nocontentsentry'):
                            continue
                        ids = sig_node['ids']
                        if not ids:
                            continue
                        anchorname = _make_anchor_name(ids, numentries)
                        reference = nodes.reference('', '', nodes.literal('', sig_node['_toc_name']), internal=True, refuri=docname, anchorname=anchorname)
                        para = addnodes.compact_paragraph('', '', reference, skip_section_number=True)
                        entry = nodes.list_item('', para)
                        *parents, _ = sig_node['_toc_parts']
                        parents = tuple(parents)
                        memo_parents[sig_node['_toc_parts']] = entry
                        if parents and parents in memo_parents:
                            root_entry = memo_parents[parents]
                            if isinstance(root_entry[-1], nodes.bullet_list):
                                root_entry[-1].append(entry)
                            else:
                                root_entry.append(nodes.bullet_list('', entry))
                            continue
                        entries.append(entry)
    if entries:
        return nodes.bullet_list('', *entries)
    return None