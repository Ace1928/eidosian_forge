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
def assign_figure_numbers(self, env: BuildEnvironment) -> List[str]:
    """Assign a figure number to each figure under a numbered toctree."""
    generated_docnames = frozenset(env.domains['std'].initial_data['labels'].keys())
    rewrite_needed = []
    assigned: Set[str] = set()
    old_fignumbers = env.toc_fignumbers
    env.toc_fignumbers = {}
    fignum_counter: Dict[str, Dict[Tuple[int, ...], int]] = {}

    def get_figtype(node: Node) -> Optional[str]:
        for domain in env.domains.values():
            figtype = domain.get_enumerable_node_type(node)
            if domain.name == 'std' and (not domain.get_numfig_title(node)):
                continue
            if figtype:
                return figtype
        return None

    def get_section_number(docname: str, section: nodes.section) -> Tuple[int, ...]:
        anchorname = '#' + section['ids'][0]
        secnumbers = env.toc_secnumbers.get(docname, {})
        if anchorname in secnumbers:
            secnum = secnumbers.get(anchorname)
        else:
            secnum = secnumbers.get('')
        return secnum or ()

    def get_next_fignumber(figtype: str, secnum: Tuple[int, ...]) -> Tuple[int, ...]:
        counter = fignum_counter.setdefault(figtype, {})
        secnum = secnum[:env.config.numfig_secnum_depth]
        counter[secnum] = counter.get(secnum, 0) + 1
        return secnum + (counter[secnum],)

    def register_fignumber(docname: str, secnum: Tuple[int, ...], figtype: str, fignode: Element) -> None:
        env.toc_fignumbers.setdefault(docname, {})
        fignumbers = env.toc_fignumbers[docname].setdefault(figtype, {})
        figure_id = fignode['ids'][0]
        fignumbers[figure_id] = get_next_fignumber(figtype, secnum)

    def _walk_doctree(docname: str, doctree: Element, secnum: Tuple[int, ...]) -> None:
        nonlocal generated_docnames
        for subnode in doctree.children:
            if isinstance(subnode, nodes.section):
                next_secnum = get_section_number(docname, subnode)
                if next_secnum:
                    _walk_doctree(docname, subnode, next_secnum)
                else:
                    _walk_doctree(docname, subnode, secnum)
            elif isinstance(subnode, addnodes.toctree):
                for _title, subdocname in subnode['entries']:
                    if url_re.match(subdocname) or subdocname == 'self':
                        continue
                    if subdocname in generated_docnames:
                        continue
                    _walk_doc(subdocname, secnum)
            elif isinstance(subnode, nodes.Element):
                figtype = get_figtype(subnode)
                if figtype and subnode['ids']:
                    register_fignumber(docname, secnum, figtype, subnode)
                _walk_doctree(docname, subnode, secnum)

    def _walk_doc(docname: str, secnum: Tuple[int, ...]) -> None:
        if docname not in assigned:
            assigned.add(docname)
            doctree = env.get_doctree(docname)
            _walk_doctree(docname, doctree, secnum)
    if env.config.numfig:
        _walk_doc(env.config.root_doc, ())
        for docname, fignums in env.toc_fignumbers.items():
            if fignums != old_fignumbers.get(docname):
                rewrite_needed.append(docname)
    return rewrite_needed