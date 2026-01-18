from os import path
from typing import Any, Dict, List, Optional, Tuple, Union
from docutils import nodes
from docutils.nodes import Node
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.toctree import TocTree
from sphinx.locale import __
from sphinx.util import logging, progress_message
from sphinx.util.console import darkgreen  # type: ignore
from sphinx.util.nodes import inline_all_toctrees
def assemble_toc_secnumbers(self) -> Dict[str, Dict[str, Tuple[int, ...]]]:
    new_secnumbers: Dict[str, Tuple[int, ...]] = {}
    for docname, secnums in self.env.toc_secnumbers.items():
        for id, secnum in secnums.items():
            alias = '%s/%s' % (docname, id)
            new_secnumbers[alias] = secnum
    return {self.config.root_doc: new_secnumbers}