import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def make_block_and_full_manager(self, texts):
    locations, block = self.make_block(texts)
    manager = groupcompress._LazyGroupContentManager(block)
    for key in sorted(texts):
        self.add_key_to_manager(key, locations, block, manager)
    return (block, manager)