import pickle
from itertools import product, zip_longest
from operator import itemgetter
from os import path
from typing import TYPE_CHECKING, Any, Dict, Iterator
from uuid import uuid4
from docutils.nodes import Node
from sphinx.transforms import SphinxTransform
def add_uids(doctree: Node, condition: Any) -> Iterator[Node]:
    """Add a unique id to every node in the `doctree` which matches the
    condition and yield the nodes.

    :param doctree:
        A :class:`docutils.nodes.document` instance.

    :param condition:
        A callable which returns either ``True`` or ``False`` for a given node.
    """
    for node in doctree.findall(condition):
        node.uid = uuid4().hex
        yield node