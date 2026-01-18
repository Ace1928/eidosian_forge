from __future__ import annotations
import logging  # isort:skip
from os.path import join
import toml
from docutils import nodes, utils
from docutils.parsers.rst.roles import set_classes
from . import PARALLEL_SAFE
from .util import _REPO_TOP
def bokeh_commit(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Link to a Bokeh Github issue.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    """
    app = inliner.document.settings.env.app
    node = _make_gh_link_node(app, rawtext, 'commit', 'commit ', 'commit', text, options)
    return ([node], [])