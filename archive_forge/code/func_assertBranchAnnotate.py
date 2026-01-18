import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def assertBranchAnnotate(self, expected, branch, path, revision_id, verbose=False, full=False, show_ids=False):
    tree = branch.repository.revision_tree(revision_id)
    to_file = StringIO()
    annotate.annotate_file_tree(tree, path, to_file, verbose=verbose, full=full, show_ids=show_ids, branch=branch)
    self.assertAnnotateEqualDiff(to_file.getvalue(), expected)