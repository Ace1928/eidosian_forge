from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def create_by_apply_delta(self, inventory_delta, new_revision_id, propagate_caches=False):
    """Create a new CHKInventory by applying inventory_delta to this one.

        See the inventory developers documentation for the theory behind
        inventory deltas.

        :param inventory_delta: The inventory delta to apply. See
            Inventory.apply_delta for details.
        :param new_revision_id: The revision id of the resulting CHKInventory.
        :param propagate_caches: If True, the caches for this inventory are
          copied to and updated for the result.
        :return: The new CHKInventory.
        """
    split = osutils.split
    result = CHKInventory(self._search_key_name)
    if propagate_caches:
        result._path_to_fileid_cache = self._path_to_fileid_cache.copy()
    search_key_func = chk_map.search_key_registry.get(self._search_key_name)
    self.id_to_entry._ensure_root()
    maximum_size = self.id_to_entry._root_node.maximum_size
    result.revision_id = new_revision_id
    result.id_to_entry = chk_map.CHKMap(self.id_to_entry._store, self.id_to_entry.key(), search_key_func=search_key_func)
    result.id_to_entry._ensure_root()
    result.id_to_entry._root_node.set_maximum_size(maximum_size)
    parent_id_basename_delta = {}
    if self.parent_id_basename_to_file_id is not None:
        result.parent_id_basename_to_file_id = chk_map.CHKMap(self.parent_id_basename_to_file_id._store, self.parent_id_basename_to_file_id.key(), search_key_func=search_key_func)
        result.parent_id_basename_to_file_id._ensure_root()
        self.parent_id_basename_to_file_id._ensure_root()
        result_p_id_root = result.parent_id_basename_to_file_id._root_node
        p_id_root = self.parent_id_basename_to_file_id._root_node
        result_p_id_root.set_maximum_size(p_id_root.maximum_size)
        result_p_id_root._key_width = p_id_root._key_width
    else:
        result.parent_id_basename_to_file_id = None
    result.root_id = self.root_id
    id_to_entry_delta = []
    inventory_delta = _check_delta_unique_ids(inventory_delta)
    inventory_delta = _check_delta_unique_old_paths(inventory_delta)
    inventory_delta = _check_delta_unique_new_paths(inventory_delta)
    inventory_delta = _check_delta_ids_match_entry(inventory_delta)
    inventory_delta = _check_delta_ids_are_valid(inventory_delta)
    inventory_delta = _check_delta_new_path_entry_both_or_None(inventory_delta)
    parents = set()
    deletes = set()
    altered = set()
    for old_path, new_path, file_id, entry in inventory_delta:
        if new_path == '':
            result.root_id = file_id
        if new_path is None:
            new_key = None
            new_value = None
            if propagate_caches:
                try:
                    del result._path_to_fileid_cache[old_path]
                except KeyError:
                    pass
            deletes.add(file_id)
        else:
            new_key = StaticTuple(file_id)
            new_value = result._entry_to_bytes(entry)
            result._path_to_fileid_cache[new_path] = file_id
            parents.add((split(new_path)[0], entry.parent_id))
        if old_path is None:
            old_key = None
        else:
            old_key = StaticTuple(file_id)
            if self.id2path(file_id) != old_path:
                raise errors.InconsistentDelta(old_path, file_id, 'Entry was at wrong other path %r.' % self.id2path(file_id))
            altered.add(file_id)
        id_to_entry_delta.append(StaticTuple(old_key, new_key, new_value))
        if result.parent_id_basename_to_file_id is not None:
            if old_path is None:
                old_key = None
            else:
                old_entry = self.get_entry(file_id)
                old_key = self._parent_id_basename_key(old_entry)
            if new_path is None:
                new_key = None
                new_value = None
            else:
                new_key = self._parent_id_basename_key(entry)
                new_value = file_id
            if old_key != new_key:
                if old_key is not None:
                    parent_id_basename_delta.setdefault(old_key, [None, None])[0] = old_key
                if new_key is not None:
                    parent_id_basename_delta.setdefault(new_key, [None, None])[1] = new_value
    for file_id in deletes:
        entry = self.get_entry(file_id)
        if entry.kind != 'directory':
            continue
        for child in entry.children.values():
            if child.file_id not in altered:
                raise errors.InconsistentDelta(self.id2path(child.file_id), child.file_id, 'Child not deleted or reparented when parent deleted.')
    result.id_to_entry.apply_delta(id_to_entry_delta)
    if parent_id_basename_delta:
        delta_list = []
        for key, (old_key, value) in parent_id_basename_delta.items():
            if value is not None:
                delta_list.append((old_key, key, value))
            else:
                delta_list.append((old_key, None, None))
        result.parent_id_basename_to_file_id.apply_delta(delta_list)
    parents.discard(('', None))
    for parent_path, parent in parents:
        try:
            if result.get_entry(parent).kind != 'directory':
                raise errors.InconsistentDelta(result.id2path(parent), parent, 'Not a directory, but given children')
        except errors.NoSuchId:
            raise errors.InconsistentDelta('<unknown>', parent, 'Parent is not present in resulting inventory.')
        if result.path2id(parent_path) != parent:
            raise errors.InconsistentDelta(parent_path, parent, 'Parent has wrong path %r.' % result.path2id(parent_path))
    return result