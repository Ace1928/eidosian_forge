import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def make_g_index_missing_parent(self):
    graph_index = self.make_g_index('missing_parent', 1, [((b'parent',), b'2 78 2 10', ([],)), ((b'tip',), b'2 78 2 10', ([(b'parent',), (b'missing-parent',)],))])
    return graph_index