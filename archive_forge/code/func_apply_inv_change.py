import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
def apply_inv_change(self, inventory_change, orig_inventory):
    orig_inventory_by_path = {}
    for file_id, path in orig_inventory.items():
        orig_inventory_by_path[path] = file_id

    def parent_id(file_id):
        try:
            parent_dir = os.path.dirname(orig_inventory[file_id])
        except:
            print(file_id)
            raise
        if parent_dir == '':
            return None
        return orig_inventory_by_path[parent_dir]

    def new_path(file_id):
        if fild_id in inventory_change:
            return inventory_change[file_id]
        else:
            parent = parent_id(file_id)
            if parent is None:
                return orig_inventory[file_id]
            dirname = new_path(parent)
            return pathjoin(dirname, os.path.basename(orig_inventory[file_id]))
    new_inventory = {}
    for file_id in orig_inventory:
        path = new_path(file_id)
        if path is None:
            continue
        new_inventory[file_id] = path
    for file_id, path in inventory_change.items():
        if file_id in orig_inventory:
            continue
        new_inventory[file_id] = path
    return new_inventory