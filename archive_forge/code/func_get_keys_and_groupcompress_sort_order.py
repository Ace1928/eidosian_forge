import itertools
from gzip import GzipFile
from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils, progress, transport, ui
from ...errors import RevisionAlreadyPresent, RevisionNotPresent
from ...tests import (TestCase, TestCaseWithMemoryTransport, TestNotApplicable,
from ...tests.http_utils import TestCaseWithWebserver
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from .. import groupcompress
from .. import knit as _mod_knit
from .. import versionedfile as versionedfile
from ..knit import cleanup_pack_knit, make_file_factory, make_pack_factory
from ..versionedfile import (ChunkedContentFactory, ConstantMapper,
from ..weave import WeaveFile, WeaveInvalidChecksum
from ..weavefile import write_weave
def get_keys_and_groupcompress_sort_order(self):
    """Get diamond test keys list, and their groupcompress sort ordering."""
    if self.key_length == 1:
        keys = [(b'merged',), (b'left',), (b'right',), (b'base',)]
        sort_order = {(b'merged',): 0, (b'left',): 1, (b'right',): 1, (b'base',): 2}
    else:
        keys = [(b'FileA', b'merged'), (b'FileA', b'left'), (b'FileA', b'right'), (b'FileA', b'base'), (b'FileB', b'merged'), (b'FileB', b'left'), (b'FileB', b'right'), (b'FileB', b'base')]
        sort_order = {(b'FileA', b'merged'): 0, (b'FileA', b'left'): 1, (b'FileA', b'right'): 1, (b'FileA', b'base'): 2, (b'FileB', b'merged'): 3, (b'FileB', b'left'): 4, (b'FileB', b'right'): 4, (b'FileB', b'base'): 5}
    return (keys, sort_order)