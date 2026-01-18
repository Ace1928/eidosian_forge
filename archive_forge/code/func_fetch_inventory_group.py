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
def fetch_inventory_group(name: str, uri: str, invs: Any, cache: Any, app: Any, now: float) -> bool:
    cache_time = now - app.config.intersphinx_cache_limit * 86400
    failures = []
    try:
        for inv in invs:
            if not inv:
                inv = posixpath.join(uri, INVENTORY_FILENAME)
            if '://' not in inv or uri not in cache or cache[uri][1] < cache_time:
                safe_inv_url = _get_safe_url(inv)
                logger.info(__('loading intersphinx inventory from %s...'), safe_inv_url)
                try:
                    invdata = fetch_inventory(app, uri, inv)
                except Exception as err:
                    failures.append(err.args)
                    continue
                if invdata:
                    cache[uri] = (name, now, invdata)
                    return True
        return False
    finally:
        if failures == []:
            pass
        elif len(failures) < len(invs):
            logger.info(__('encountered some issues with some of the inventories, but they had working alternatives:'))
            for fail in failures:
                logger.info(*fail)
        else:
            issues = '\n'.join([f[0] % f[1:] for f in failures])
            logger.warning(__('failed to reach any of the inventories with the following issues:') + '\n' + issues)