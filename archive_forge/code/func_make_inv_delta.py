from breezy import errors, revision
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def make_inv_delta(self, old, new):
    """Make an inventory delta from two inventories."""
    by_id = getattr(old, '_byid', None)
    if by_id is None:
        old_ids = {entry.file_id for entry in old.iter_just_entries()}
    else:
        old_ids = set(by_id)
    by_id = getattr(new, '_byid', None)
    if by_id is None:
        new_ids = {entry.file_id for entry in new.iter_just_entries()}
    else:
        new_ids = set(by_id)
    adds = new_ids - old_ids
    deletes = old_ids - new_ids
    common = old_ids.intersection(new_ids)
    delta = []
    for file_id in deletes:
        delta.append((old.id2path(file_id), None, file_id, None))
    for file_id in adds:
        delta.append((None, new.id2path(file_id), file_id, new.get_entry(file_id)))
    for file_id in common:
        if old.get_entry(file_id) != new.get_entry(file_id):
            delta.append((old.id2path(file_id), new.id2path(file_id), file_id, new[file_id]))
    return delta