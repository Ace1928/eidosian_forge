import functools
import os
import re
import sys
import warnings
from io import StringIO
from typing import IO, Any, Dict, Generator, List, Optional, Pattern
from xml.etree import ElementTree
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives, roles
from sphinx import application, locale
from sphinx.pycode import ModuleAnalyzer
from sphinx.testing.path import path
from sphinx.util.osutil import relpath
def assert_node(node: Node, cls: Any=None, xpath: str='', **kwargs: Any) -> None:
    if cls:
        if isinstance(cls, list):
            assert_node(node, cls[0], xpath=xpath, **kwargs)
            if cls[1:]:
                if isinstance(cls[1], tuple):
                    assert_node(node, cls[1], xpath=xpath, **kwargs)
                else:
                    assert isinstance(node, nodes.Element), 'The node%s does not have any children' % xpath
                    assert len(node) == 1, 'The node%s has %d child nodes, not one' % (xpath, len(node))
                    assert_node(node[0], cls[1:], xpath=xpath + '[0]', **kwargs)
        elif isinstance(cls, tuple):
            assert isinstance(node, (list, nodes.Element)), 'The node%s does not have any items' % xpath
            assert len(node) == len(cls), 'The node%s has %d child nodes, not %r' % (xpath, len(node), len(cls))
            for i, nodecls in enumerate(cls):
                path = xpath + '[%d]' % i
                assert_node(node[i], nodecls, xpath=path, **kwargs)
        elif isinstance(cls, str):
            assert node == cls, 'The node %r is not %r: %r' % (xpath, cls, node)
        else:
            assert isinstance(node, cls), 'The node%s is not subclass of %r: %r' % (xpath, cls, node)
    if kwargs:
        assert isinstance(node, nodes.Element), 'The node%s does not have any attributes' % xpath
        for key, value in kwargs.items():
            assert key in node, 'The node%s does not have %r attribute: %r' % (xpath, key, node)
            assert node[key] == value, 'The node%s[%s] is not %r: %r' % (xpath, key, value, node[key])