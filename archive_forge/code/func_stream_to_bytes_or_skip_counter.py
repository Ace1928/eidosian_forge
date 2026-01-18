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
def stream_to_bytes_or_skip_counter(self, skipped_records, full_texts, stream):
    """Convert a stream to a bytes iterator.

        :param skipped_records: A list with one element to increment when a
            record is skipped.
        :param full_texts: A dict from key->fulltext representation, for
            checking chunked or fulltext stored records.
        :param stream: A record_stream.
        :return: An iterator over the bytes of each record.
        """
    for record in stream:
        if record.storage_kind in ('chunked', 'fulltext'):
            skipped_records[0] += 1
            self.assertRecordHasContent(record, full_texts[record.key])
        else:
            yield record.get_bytes_as(record.storage_kind)