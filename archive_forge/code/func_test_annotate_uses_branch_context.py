import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def test_annotate_uses_branch_context(self):
    """Dotted revnos should use the Branch context.

        When annotating a non-mainline revision, the annotation should still
        use dotted revnos from the mainline.
        """
    builder = self.create_deeply_merged_trees()
    self.assertBranchAnnotate('1     joe@foo | first\n1.1.1 barry@f | third\n1.2.1 jerry@f | fourth\n1.3.1 george@ | fifth\n              | sixth\n', builder.get_branch(), 'a', b'rev-1_3_1', verbose=False, full=False)