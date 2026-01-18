import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
def show_changes(self, merger):
    """Show the changes that this operation specifies."""
    tree_merger = merger.make_merger()
    tt = tree_merger.make_preview_transform()
    tt.finalize()