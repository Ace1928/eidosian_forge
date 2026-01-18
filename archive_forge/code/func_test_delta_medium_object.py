import binascii
import os
import re
import shutil
import tempfile
from dulwich.tests import SkipTest
from ...objects import Blob
from ...pack import write_pack
from ..test_pack import PackTests, a_sha, pack1_sha
from .utils import require_git_version, run_git_or_fail
def test_delta_medium_object(self):
    with self.get_pack(pack1_sha) as orig_pack:
        orig_blob = orig_pack[a_sha]
        new_blob = Blob()
        new_blob.data = orig_blob.data + b'x' * 2 ** 20
        new_blob_2 = Blob()
        new_blob_2.data = new_blob.data + b'y'
        all_to_pack = [*list(orig_pack.pack_tuples()), (new_blob, None), (new_blob_2, None)]
        pack_path = os.path.join(self._tempdir, 'pack_with_deltas')
        write_pack(pack_path, all_to_pack, deltify=True)
    output = run_git_or_fail(['verify-pack', '-v', pack_path])
    self.assertEqual({x[0].id for x in all_to_pack}, _git_verify_pack_object_list(output))
    got_non_delta = int(_NON_DELTA_RE.search(output).group('non_delta'))
    self.assertEqual(3, got_non_delta, 'Expected 3 non-delta objects, got %d' % got_non_delta)
    self.assertIn(b'chain length = 2', output)