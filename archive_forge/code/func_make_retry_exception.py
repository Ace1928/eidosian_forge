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
def make_retry_exception(self):
    try:
        raise _TestException('foobar')
    except _TestException as e:
        retry_exc = pack_repo.RetryWithNewPacks(None, reload_occurred=False, exc_info=sys.exc_info())
    return retry_exc