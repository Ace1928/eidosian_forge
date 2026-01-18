import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def create_merged_trees(self):
    """create 2 trees with merges between them.

        rev-1 --+
         |      |
        rev-2  rev-1_1_1
         |      |
         +------+
         |
        rev-3
        """
    builder = self.make_branch_builder('branch')
    builder.start_series()
    self.addCleanup(builder.finish_series)
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'first\n'))], timestamp=1166046000.0, timezone=0, committer='joe@foo.com', revision_id=b'rev-1')
    builder.build_snapshot([b'rev-1'], [('modify', ('a', b'first\nsecond\n'))], timestamp=1166046001.0, timezone=0, committer='joe@foo.com', revision_id=b'rev-2')
    builder.build_snapshot([b'rev-1'], [('modify', ('a', b'first\nthird\n'))], timestamp=1166046002.0, timezone=0, committer='barry@foo.com', revision_id=b'rev-1_1_1')
    builder.build_snapshot([b'rev-2', b'rev-1_1_1'], [('modify', ('a', b'first\nsecond\nthird\n'))], timestamp=1166046003.0, timezone=0, committer='sal@foo.com', revision_id=b'rev-3')
    return builder