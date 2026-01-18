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
def test_get_record_stream_ordered_fulltexts(self):
    basis, test = self.get_basis_and_test_knit()
    key = (b'foo',)
    key_basis = (b'bar',)
    key_basis_2 = (b'quux',)
    key_missing = (b'missing',)
    test.add_lines(key, (key_basis,), [b'foo\n'])
    basis.add_lines(key_basis, (key_basis_2,), [b'foo\n', b'bar\n'])
    basis.add_lines(key_basis_2, (), [b'quux\n'])
    basis.calls = []
    records = list(test.get_record_stream([key, key_basis, key_missing, key_basis_2], 'topological', True))
    self.assertEqual(4, len(records))
    results = []
    for record in records:
        self.assertSubset([record.key], (key_basis, key_missing, key_basis_2, key))
        if record.key == key_missing:
            self.assertIsInstance(record, AbsentContentFactory)
        else:
            results.append((record.key, record.sha1, record.storage_kind, record.get_bytes_as('fulltext')))
    calls = list(basis.calls)
    order = [record[0] for record in results]
    self.assertEqual([key_basis_2, key_basis, key], order)
    for result in results:
        if result[0] == key:
            source = test
        else:
            source = basis
        record = next(source.get_record_stream([result[0]], 'unordered', True))
        self.assertEqual(record.key, result[0])
        self.assertEqual(record.sha1, result[1])
        self.assertEqual(record.get_bytes_as('fulltext'), result[3])
    self.assertEqual(2, len(calls))
    self.assertEqual(('get_parent_map', {key_basis, key_basis_2, key_missing}), calls[0])
    self.assertIn(calls[1], [('get_record_stream', [key_basis_2, key_basis], 'topological', True), ('get_record_stream', [key_basis, key_basis_2], 'topological', True)])