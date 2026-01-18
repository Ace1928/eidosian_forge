import subprocess
import tempfile
from ... import errors
from ... import revision as _mod_revision
from ...config import ListOption, Option, bool_from_store, int_from_store
from ...email_message import EmailMessage
from ...smtp_connection import SMTPConnection
def get_diff(self):
    """Add the diff from the commit to the output.

        If the diff has more than difflimit lines, it will be skipped.
        """
    difflimit = self.difflimit()
    if not difflimit:
        return
    from ...diff import show_diff_trees
    revid_new = self.revision.revision_id
    if self.revision.parent_ids:
        revid_old = self.revision.parent_ids[0]
        tree_new, tree_old = self.repository.revision_trees((revid_new, revid_old))
    else:
        revid_old = _mod_revision.NULL_REVISION
        tree_new = self.repository.revision_tree(revid_new)
        tree_old = self.repository.revision_tree(revid_old)
    from io import BytesIO
    diff_content = BytesIO()
    diff_options = self.config.get('post_commit_diffoptions')
    show_diff_trees(tree_old, tree_new, diff_content, None, diff_options)
    numlines = diff_content.getvalue().count(b'\n') + 1
    if numlines <= difflimit:
        return diff_content.getvalue()
    else:
        return '\nDiff too large for email (%d lines, the limit is %d).\n' % (numlines, difflimit)