import re
import sys
from typing import Any, Dict, List, Tuple
from docutils import nodes, utils
from docutils.nodes import Node, system_message
from docutils.parsers.rst.states import Inliner
import sphinx
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import logging, rst
from sphinx.util.nodes import split_explicit_title
from sphinx.util.typing import RoleFunction
def setup_link_roles(app: Sphinx) -> None:
    for name, (base_url, caption) in app.config.extlinks.items():
        app.add_role(name, make_link_role(name, base_url, caption))