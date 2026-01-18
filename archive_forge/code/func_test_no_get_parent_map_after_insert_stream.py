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
def test_no_get_parent_map_after_insert_stream(self):
    self.setup_smart_server_with_call_log()
    if isinstance(self.branch_format_from, bzrbranch.BranchReferenceFormat):
        raise tests.TestSkipped("BranchBuilder can't make reference branches.")
    try:
        builder = self.make_from_branch_builder('local')
    except (errors.TransportNotPossible, errors.UninitializableFormat):
        raise tests.TestNotApplicable('format not directly constructable')
    builder.start_series()
    first = builder.build_snapshot(None, [('add', ('', None, 'directory', ''))])
    second = builder.build_snapshot([first], [])
    third = builder.build_snapshot([second], [])
    fourth = builder.build_snapshot([third], [])
    builder.finish_series()
    local = branch.Branch.open(self.get_vfs_only_url('local'))
    remote_bzrdir = local.controldir.sprout(self.get_url('remote'), revision_id=third)
    remote = remote_bzrdir.open_branch()
    if not remote.repository._format.supports_full_versioned_files:
        raise tests.TestNotApplicable('remote is not a VersionedFile repository')
    self.reset_smart_call_log()
    self.disableOptimisticGetParentMap()
    self.assertFalse(local.is_locked())
    local.push(remote)
    hpss_call_names = [item.call.method for item in self.hpss_calls]
    self.assertIn(b'Repository.insert_stream_1.19', hpss_call_names)
    insert_stream_idx = hpss_call_names.index(b'Repository.insert_stream_1.19')
    calls_after_insert_stream = hpss_call_names[insert_stream_idx:]
    bzr_core_trace = Equals([b'Repository.insert_stream_1.19', b'Repository.insert_stream_1.19', b'Branch.set_last_revision_info', b'Branch.unlock'])
    bzr_loom_trace = Equals([b'Repository.insert_stream_1.19', b'Repository.insert_stream_1.19', b'Branch.set_last_revision_info', b'get', b'Branch.unlock'])
    self.assertThat(calls_after_insert_stream, MatchesAny(bzr_core_trace, bzr_loom_trace))