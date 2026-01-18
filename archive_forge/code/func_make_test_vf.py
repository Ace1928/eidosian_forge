import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def make_test_vf(self):
    t = self.get_transport('.')
    t.ensure_base()
    factory = groupcompress.make_pack_factory(graph=True, delta=False, keylength=1, inconsistency_fatal=True)
    vf = factory(t)
    self.addCleanup(groupcompress.cleanup_pack_group, vf)
    return vf