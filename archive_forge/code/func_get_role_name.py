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
def get_role_name(self, name: str) -> Optional[Tuple[str, str]]:
    names = name.split(':')
    if len(names) == 1:
        default_domain = self.env.temp_data.get('default_domain')
        domain = default_domain.name if default_domain else None
        role = names[0]
    elif len(names) == 2:
        domain = names[0]
        role = names[1]
    else:
        return None
    if domain and self.is_existent_role(domain, role):
        return (domain, role)
    elif self.is_existent_role('std', role):
        return ('std', role)
    else:
        return None