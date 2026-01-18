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
def get_next_fignumber(figtype: str, secnum: Tuple[int, ...]) -> Tuple[int, ...]:
    counter = fignum_counter.setdefault(figtype, {})
    secnum = secnum[:env.config.numfig_secnum_depth]
    counter[secnum] = counter.get(secnum, 0) + 1
    return secnum + (counter[secnum],)