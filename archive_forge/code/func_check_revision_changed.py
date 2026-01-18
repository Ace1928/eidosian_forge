from ... import osutils, trace, ui
from ...errors import BzrError
from .rebase import (CommitBuilderRevisionRewriter, generate_transpose_plan,
def check_revision_changed(oldrev, newrev):
    """Check if two revisions are different. This is exactly the same
    as Revision.equals() except that it does not check the revision_id."""
    if newrev.inventory_sha1 != oldrev.inventory_sha1 or newrev.timestamp != oldrev.timestamp or newrev.message != oldrev.message or (newrev.timezone != oldrev.timezone) or (newrev.committer != oldrev.committer) or (newrev.properties != oldrev.properties):
        raise UpgradeChangesContent(oldrev.revision_id)