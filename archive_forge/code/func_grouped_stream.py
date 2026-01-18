import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
@staticmethod
def grouped_stream(revision_ids, first_parents=()):
    parents = first_parents
    for revision_id in revision_ids:
        key = (revision_id,)
        record = versionedfile.FulltextContentFactory(key, parents, None, b'some content that is\nidentical except for\nrevision_id:%s\n' % (revision_id,))
        yield record
        parents = (key,)