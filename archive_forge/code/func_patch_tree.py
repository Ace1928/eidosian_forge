import errno
import os
import sys
import tempfile
from subprocess import PIPE, Popen
from .errors import BzrError, NoDiff3
from .textfile import check_text_path
def patch_tree(tree, patches, strip=0, reverse=False, dry_run=False, quiet=False, out=None):
    """Apply a patch to a tree.

    Args:
      tree: A MutableTree object
      patches: list of patches as bytes
      strip: Strip X segments of paths
      reverse: Apply reversal of patch
      dry_run: Dry run
    """
    return run_patch(tree.basedir, patches, strip, reverse, dry_run, quiet, out=out)