import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, Set, cast
from docutils import nodes
from docutils.nodes import Element
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util import inspect, typing
def insert_field_list(node: Element) -> nodes.field_list:
    field_list = nodes.field_list()
    desc = [n for n in node if isinstance(n, addnodes.desc)]
    if desc:
        index = node.index(desc[0])
        node.insert(index - 1, [field_list])
    else:
        node += field_list
    return field_list