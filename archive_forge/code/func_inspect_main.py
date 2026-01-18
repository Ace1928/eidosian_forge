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
def inspect_main(argv: List[str]) -> None:
    """Debug functionality to print out an inventory"""
    if len(argv) < 1:
        print('Print out an inventory file.\nError: must specify local path or URL to an inventory file.', file=sys.stderr)
        sys.exit(1)

    class MockConfig:
        intersphinx_timeout: Optional[int] = None
        tls_verify = False
        user_agent = None

    class MockApp:
        srcdir = ''
        config = MockConfig()

        def warn(self, msg: str) -> None:
            print(msg, file=sys.stderr)
    try:
        filename = argv[0]
        invdata = fetch_inventory(MockApp(), '', filename)
        for key in sorted(invdata or {}):
            print(key)
            for entry, einfo in sorted(invdata[key].items()):
                print('\t%-40s %s%s' % (entry, '%-40s: ' % einfo[3] if einfo[3] != '-' else '', einfo[2]))
    except ValueError as exc:
        print(exc.args[0] % exc.args[1:])
    except Exception as exc:
        print('Unknown error: %r' % exc)