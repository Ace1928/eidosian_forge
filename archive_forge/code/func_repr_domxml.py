import re
import unicodedata
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple, Type,
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging
def repr_domxml(node: Node, length: int=80) -> str:
    """
    return DOM XML representation of the specified node like:
    '<paragraph translatable="False"><inline classes="versionmodified">New in version...'

    :param nodes.Node node: target node
    :param int length:
       length of return value to be striped. if false-value is specified, repr_domxml
       returns full of DOM XML representation.
    :return: DOM XML representation
    """
    try:
        text = node.asdom().toxml()
    except Exception:
        text = str(node)
    if length and len(text) > length:
        text = text[:length] + '...'
    return text