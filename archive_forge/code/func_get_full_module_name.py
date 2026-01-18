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
def get_full_module_name(node: Node) -> str:
    """
    Return full module dotted path like: 'docutils.nodes.paragraph'

    :param nodes.Node node: target node
    :return: full module dotted path
    """
    return '{}.{}'.format(node.__module__, node.__class__.__name__)