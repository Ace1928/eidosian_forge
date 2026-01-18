import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def small_size_stream():
    for record in vf.get_record_stream(keys, 'groupcompress', False):
        record._manager._full_enough_block_size = record._manager._block._content_length
        yield record