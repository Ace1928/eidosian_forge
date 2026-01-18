from __future__ import annotations
import logging  # isort:skip
from os.path import join
import toml
from docutils import nodes, utils
from docutils.parsers.rst.roles import set_classes
from . import PARALLEL_SAFE
from .util import _REPO_TOP
def bokeh_requires(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Provide the list of required package dependencies for Bokeh.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    """
    pyproject = toml.load(join(_REPO_TOP, 'pyproject.toml'))
    node = nodes.bullet_list()
    for dep in pyproject['project']['dependencies']:
        node += nodes.list_item('', nodes.Text(dep))
    return ([node], [])