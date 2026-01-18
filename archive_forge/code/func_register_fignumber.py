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
def register_fignumber(docname: str, secnum: Tuple[int, ...], figtype: str, fignode: Element) -> None:
    env.toc_fignumbers.setdefault(docname, {})
    fignumbers = env.toc_fignumbers[docname].setdefault(figtype, {})
    figure_id = fignode['ids'][0]
    fignumbers[figure_id] = get_next_fignumber(figtype, secnum)