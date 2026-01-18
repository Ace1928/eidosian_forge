from breezy import tests
from breezy.revision import NULL_REVISION
from breezy.tests import per_workingtree
def make_branch_deleting_dir(self, relpath=None):
    if relpath is None:
        relpath = 'trunk'
    builder = self.make_branch_builder(relpath)
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', ''))], revision_id=b'1')
    builder.build_snapshot([b'1'], [('add', ('dir', b'dir-id', 'directory', '')), ('add', ('file', b'file-id', 'file', b'trunk content\n'))], revision_id=b'2')
    builder.build_snapshot([b'2'], [('unversion', 'dir')], revision_id=b'3')
    builder.finish_series()
    return builder.get_branch()