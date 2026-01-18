from breezy import branch, errors, tests
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.transport import FileExists, NoSuchFile
def test_create_clone_of_multiple_roots(self):
    try:
        builder = self.make_branch_builder('local')
    except (errors.TransportNotPossible, errors.UninitializableFormat):
        raise tests.TestNotApplicable('format not directly constructable')
    builder.start_series()
    rev1 = builder.build_snapshot(None, [('add', ('', None, 'directory', ''))])
    rev2 = builder.build_snapshot([rev1], [])
    other = builder.build_snapshot(None, [('add', ('', None, 'directory', ''))])
    rev3 = builder.build_snapshot([rev2, other], [])
    builder.finish_series()
    local = builder.get_branch()
    local.controldir.clone(self.get_url('remote'), revision_id=rev3)