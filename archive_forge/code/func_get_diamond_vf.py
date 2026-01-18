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
def get_diamond_vf(f, trailing_eol=True, left_only=False):
    """Get a diamond graph to exercise deltas and merges.

    :param trailing_eol: If True end the last line with 
.
    """
    parents = {b'origin': (), b'base': ((b'origin',),), b'left': ((b'base',),), b'right': ((b'base',),), b'merged': ((b'left',), (b'right',))}
    if trailing_eol:
        last_char = b'\n'
    else:
        last_char = b''
    f.add_lines(b'origin', [], [b'origin' + last_char])
    f.add_lines(b'base', [b'origin'], [b'base' + last_char])
    f.add_lines(b'left', [b'base'], [b'base\n', b'left' + last_char])
    if not left_only:
        f.add_lines(b'right', [b'base'], [b'base\n', b'right' + last_char])
        f.add_lines(b'merged', [b'left', b'right'], [b'base\n', b'left\n', b'right\n', b'merged' + last_char])
    return (f, parents)