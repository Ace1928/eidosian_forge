from io import BytesIO
from testtools.matchers import Equals, MatchesAny
from ... import branch, check, controldir, errors, push, tests
from ...branch import BindingUnsupported, Branch
from ...bzr import branch as bzrbranch
from ...bzr import vf_repository
from ...bzr.smart.repository import SmartServerRepositoryGetParentMap
from ...controldir import ControlDir
from ...revision import NULL_REVISION
from .. import test_server
from . import TestCaseWithInterBranch
def disableOptimisticGetParentMap(self):
    self.overrideAttr(vf_repository.InterVersionedFileRepository, '_walk_to_common_revisions_batch_size', 1)
    self.overrideAttr(SmartServerRepositoryGetParentMap, 'no_extra_results', True)