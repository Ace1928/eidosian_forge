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
def make_link_role(name: str, base_url: str, caption: str) -> RoleFunction:
    try:
        base_url % 'dummy'
    except (TypeError, ValueError):
        logger.warning(__("extlinks: Sphinx-6.0 will require base URL to contain exactly one '%s' and all other '%' need to be escaped as '%%'."))
        base_url = base_url.replace('%', '%%') + '%s'
    if caption is not None:
        try:
            caption % 'dummy'
        except (TypeError, ValueError):
            logger.warning(__("extlinks: Sphinx-6.0 will require a caption string to contain exactly one '%s' and all other '%' need to be escaped as '%%'."))
            caption = caption.replace('%', '%%') + '%s'

    def role(typ: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options: Dict={}, content: List[str]=[]) -> Tuple[List[Node], List[system_message]]:
        text = utils.unescape(text)
        has_explicit_title, title, part = split_explicit_title(text)
        full_url = base_url % part
        if not has_explicit_title:
            if caption is None:
                title = full_url
            else:
                title = caption % part
        pnode = nodes.reference(title, title, internal=False, refuri=full_url)
        return ([pnode], [])
    return role