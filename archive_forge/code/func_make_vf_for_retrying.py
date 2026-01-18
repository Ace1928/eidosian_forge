import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
def make_vf_for_retrying(self):
    """Create 3 packs and a reload function.

        Originally, 2 pack files will have the data, but one will be missing.
        And then the third will be used in place of the first two if reload()
        is called.

        :return: (versioned_file, reload_counter)
            versioned_file  a KnitVersionedFiles using the packs for access
        """
    builder = self.make_branch_builder('.', format='1.9')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\nrev 1\n'))], revision_id=b'rev-1')
    builder.build_snapshot([b'rev-1'], [('modify', ('file', b'content\nrev 2\n'))], revision_id=b'rev-2')
    builder.build_snapshot([b'rev-2'], [('modify', ('file', b'content\nrev 3\n'))], revision_id=b'rev-3')
    builder.finish_series()
    b = builder.get_branch()
    b.lock_write()
    self.addCleanup(b.unlock)
    repo = b.repository
    collection = repo._pack_collection
    collection.ensure_loaded()
    orig_packs = collection.packs
    packer = knitpack_repo.KnitPacker(collection, orig_packs, '.testpack')
    new_pack = packer.pack()
    collection.reset()
    repo.refresh_data()
    vf = repo.revisions
    new_index = new_pack.revision_index
    access_tuple = new_pack.access_tuple()
    reload_counter = [0, 0, 0]

    def reload():
        reload_counter[0] += 1
        if reload_counter[1] > 0:
            reload_counter[2] += 1
            return False
        reload_counter[1] += 1
        vf._index._graph_index._indices[:] = [new_index]
        vf._access._indices.clear()
        vf._access._indices[new_index] = access_tuple
        return True
    trans, name = orig_packs[1].access_tuple()
    trans.delete(name)
    vf._access._reload_func = reload
    return (vf, reload_counter)