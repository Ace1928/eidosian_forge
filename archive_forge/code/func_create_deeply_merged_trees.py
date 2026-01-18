import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def create_deeply_merged_trees(self):
    """Create some trees with a more complex merge history.

        rev-1 --+
         |      |
        rev-2  rev-1_1_1 --+
         |      |          |
         +------+          |
         |      |          |
        rev-3  rev-1_1_2  rev-1_2_1 ------+
         |      |          |              |
         +------+          |              |
         |                 |              |
        rev-4             rev-1_2_2  rev-1_3_1
         |                 |              |
         +-----------------+              |
         |                                |
        rev-5                             |
         |                                |
         +--------------------------------+
         |
        rev-6
        """
    builder = self.create_merged_trees()
    builder.build_snapshot([b'rev-1_1_1'], [], revision_id=b'rev-1_1_2')
    builder.build_snapshot([b'rev-3', b'rev-1_1_2'], [], revision_id=b'rev-4')
    builder.build_snapshot([b'rev-1_1_1'], [('modify', ('a', b'first\nthird\nfourth\n'))], timestamp=1166046003.0, timezone=0, committer='jerry@foo.com', revision_id=b'rev-1_2_1')
    builder.build_snapshot([b'rev-1_2_1'], [], timestamp=1166046004.0, timezone=0, committer='jerry@foo.com', revision_id=b'rev-1_2_2')
    builder.build_snapshot([b'rev-4', b'rev-1_2_2'], [('modify', ('a', b'first\nsecond\nthird\nfourth\n'))], timestamp=1166046004.0, timezone=0, committer='jerry@foo.com', revision_id=b'rev-5')
    builder.build_snapshot([b'rev-1_2_1'], [('modify', ('a', b'first\nthird\nfourth\nfifth\nsixth\n'))], timestamp=1166046005.0, timezone=0, committer='george@foo.com', revision_id=b'rev-1_3_1')
    builder.build_snapshot([b'rev-5', b'rev-1_3_1'], [('modify', ('a', b'first\nsecond\nthird\nfourth\nfifth\nsixth\n'))], revision_id=b'rev-6')
    return builder