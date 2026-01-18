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
def test__group_keys_for_io(self):
    ft_detail = ('fulltext', False)
    ld_detail = ('line-delta', False)
    f_a = (b'f', b'a')
    f_b = (b'f', b'b')
    f_c = (b'f', b'c')
    g_a = (b'g', b'a')
    g_b = (b'g', b'b')
    g_c = (b'g', b'c')
    positions = {f_a: (ft_detail, (f_a, 0, 100), None), f_b: (ld_detail, (f_b, 100, 21), f_a), f_c: (ld_detail, (f_c, 180, 15), f_b), g_a: (ft_detail, (g_a, 121, 35), None), g_b: (ld_detail, (g_b, 156, 12), g_a), g_c: (ld_detail, (g_c, 195, 13), g_a)}
    self.assertGroupKeysForIo([([f_a], set())], [f_a], [], positions)
    self.assertGroupKeysForIo([([f_a], {f_a})], [f_a], [f_a], positions)
    self.assertGroupKeysForIo([([f_a, f_b], set())], [f_a, f_b], [], positions)
    self.assertGroupKeysForIo([([f_a, f_b], {f_b})], [f_a, f_b], [f_b], positions)
    self.assertGroupKeysForIo([([f_a, f_b, g_a, g_b], set())], [f_a, g_a, f_b, g_b], [], positions)
    self.assertGroupKeysForIo([([f_a, f_b, g_a, g_b], set())], [f_a, g_a, f_b, g_b], [], positions, _min_buffer_size=150)
    self.assertGroupKeysForIo([([f_a, f_b], set()), ([g_a, g_b], set())], [f_a, g_a, f_b, g_b], [], positions, _min_buffer_size=100)
    self.assertGroupKeysForIo([([f_c], set()), ([g_b], set())], [f_c, g_b], [], positions, _min_buffer_size=125)
    self.assertGroupKeysForIo([([g_b, f_c], set())], [g_b, f_c], [], positions, _min_buffer_size=125)