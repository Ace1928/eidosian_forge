from __future__ import annotations
import logging  # isort:skip
from os.path import join
import toml
from docutils import nodes, utils
from docutils.parsers.rst.roles import set_classes
from . import PARALLEL_SAFE
from .util import _REPO_TOP
def bokeh_tree(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Link to a URL in the Bokeh GitHub tree, pointing to appropriate tags
    for releases, or to main otherwise.

    The link text is simply the URL path supplied, so typical usage might
    look like:

    .. code-block:: none

        All of the examples are located in the :bokeh-tree:`examples`
        subdirectory of your Bokeh checkout.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    """
    app = inliner.document.settings.env.app
    tag = app.env.config['version']
    if '-' in tag:
        tag = 'main'
    url = f'{BOKEH_GH}/tree/{tag}/{text}'
    options = options or {}
    set_classes(options)
    node = nodes.reference(rawtext, text, refuri=url, **options)
    return ([node], [])