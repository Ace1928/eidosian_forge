import concurrent.futures
import functools
import posixpath
import re
import sys
import time
from os import path
from types import ModuleType
from typing import IO, Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urlsplit, urlunsplit
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.utils import Reporter, relative_path
import sphinx
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders.html import INVENTORY_FILENAME
from sphinx.config import Config
from sphinx.domains import Domain
from sphinx.environment import BuildEnvironment
from sphinx.errors import ExtensionError
from sphinx.locale import _, __
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging, requests
from sphinx.util.docutils import CustomReSTDispatcher, SphinxRole
from sphinx.util.inventory import InventoryFile
from sphinx.util.typing import Inventory, InventoryItem, RoleFunction
def fetch_inventory(app: Sphinx, uri: str, inv: Any) -> Any:
    """Fetch, parse and return an intersphinx inventory file."""
    localuri = '://' not in uri
    if not localuri:
        uri = _strip_basic_auth(uri)
    try:
        if '://' in inv:
            f = _read_from_url(inv, config=app.config)
        else:
            f = open(path.join(app.srcdir, inv), 'rb')
    except Exception as err:
        err.args = ('intersphinx inventory %r not fetchable due to %s: %s', inv, err.__class__, str(err))
        raise
    try:
        if hasattr(f, 'url'):
            newinv = f.url
            if inv != newinv:
                logger.info(__('intersphinx inventory has moved: %s -> %s'), inv, newinv)
                if uri in (inv, path.dirname(inv), path.dirname(inv) + '/'):
                    uri = path.dirname(newinv)
        with f:
            try:
                join = path.join if localuri else posixpath.join
                invdata = InventoryFile.load(f, uri, join)
            except ValueError as exc:
                raise ValueError('unknown or unsupported inventory version: %r' % exc) from exc
    except Exception as err:
        err.args = ('intersphinx inventory %r not readable due to %s: %s', inv, err.__class__.__name__, str(err))
        raise
    else:
        return invdata