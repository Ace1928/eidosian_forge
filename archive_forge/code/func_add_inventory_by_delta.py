import time
from .. import controldir, debug, errors, osutils
from .. import revision as _mod_revision
from .. import trace, ui
from ..bzr import chk_map, chk_serializer
from ..bzr import index as _mod_index
from ..bzr import inventory, pack, versionedfile
from ..bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from ..bzr.groupcompress import GroupCompressVersionedFiles, _GCGraphIndex
from ..bzr.vf_repository import StreamSource
from .pack_repo import (NewPack, Pack, PackCommitBuilder, Packer,
from .static_tuple import StaticTuple
def add_inventory_by_delta(self, basis_revision_id, delta, new_revision_id, parents, basis_inv=None, propagate_caches=False):
    """Add a new inventory expressed as a delta against another revision.

        :param basis_revision_id: The inventory id the delta was created
            against.
        :param delta: The inventory delta (see Inventory.apply_delta for
            details).
        :param new_revision_id: The revision id that the inventory is being
            added for.
        :param parents: The revision ids of the parents that revision_id is
            known to have and are in the repository already. These are supplied
            for repositories that depend on the inventory graph for revision
            graph access, as well as for those that pun ancestry with delta
            compression.
        :param basis_inv: The basis inventory if it is already known,
            otherwise None.
        :param propagate_caches: If True, the caches for this inventory are
          copied to and updated for the result if possible.

        :returns: (validator, new_inv)
            The validator(which is a sha1 digest, though what is sha'd is
            repository format specific) of the serialized inventory, and the
            resulting inventory.
        """
    if not self.is_in_write_group():
        raise AssertionError('{!r} not in write group'.format(self))
    _mod_revision.check_not_reserved_id(new_revision_id)
    basis_tree = None
    if basis_inv is None or not isinstance(basis_inv, inventory.CHKInventory):
        if basis_revision_id == _mod_revision.NULL_REVISION:
            new_inv = self._create_inv_from_null(delta, new_revision_id)
            if new_inv.root_id is None:
                raise errors.RootMissing()
            inv_lines = new_inv.to_lines()
            return (self._inventory_add_lines(new_revision_id, parents, inv_lines, check_content=False), new_inv)
        else:
            basis_tree = self.revision_tree(basis_revision_id)
            basis_tree.lock_read()
            basis_inv = basis_tree.root_inventory
    try:
        result = basis_inv.create_by_apply_delta(delta, new_revision_id, propagate_caches=propagate_caches)
        inv_lines = result.to_lines()
        return (self._inventory_add_lines(new_revision_id, parents, inv_lines, check_content=False), result)
    finally:
        if basis_tree is not None:
            basis_tree.unlock()