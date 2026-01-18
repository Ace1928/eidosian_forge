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
def iter_merge3(retval):
    retval['text_conflicts'] = False
    if base_marker and self.reprocess:
        raise CantReprocessAndShowBase()
    lines = list(m3.merge_lines(name_a=b'TREE', name_b=b'MERGE-SOURCE', name_base=b'BASE-REVISION', start_marker=start_marker, base_marker=base_marker, reprocess=self.reprocess))
    for line in lines:
        if line.startswith(start_marker):
            retval['text_conflicts'] = True
            yield line.replace(start_marker, b'<' * 7)
        else:
            yield line