import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def dump_file(self, temp_dir, name, tree, path):
    out_path = osutils.pathjoin(temp_dir, name)
    with open(out_path, 'wb') as out_file:
        in_file = tree.get_file(path)
        for line in in_file:
            out_file.write(line)
    return out_path