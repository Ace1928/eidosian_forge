import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def group_compress_implementation_scenarios():
    scenarios = [('python', {'compressor': groupcompress.PythonGroupCompressor})]
    if compiled_groupcompress_feature.available():
        scenarios.append(('C', {'compressor': groupcompress.PyrexGroupCompressor}))
    return scenarios