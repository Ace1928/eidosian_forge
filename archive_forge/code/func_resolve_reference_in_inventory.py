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
def resolve_reference_in_inventory(env: BuildEnvironment, inv_name: str, node: pending_xref, contnode: TextElement) -> Optional[Element]:
    """Attempt to resolve a missing reference via intersphinx references.

    Resolution is tried in the given inventory with the target as is.

    Requires ``inventory_exists(env, inv_name)``.
    """
    assert inventory_exists(env, inv_name)
    return _resolve_reference(env, inv_name, InventoryAdapter(env).named_inventory[inv_name], False, node, contnode)