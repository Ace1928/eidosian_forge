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
def get_section_number(docname: str, section: nodes.section) -> Tuple[int, ...]:
    anchorname = '#' + section['ids'][0]
    secnumbers = env.toc_secnumbers.get(docname, {})
    if anchorname in secnumbers:
        secnum = secnumbers.get(anchorname)
    else:
        secnum = secnumbers.get('')
    return secnum or ()