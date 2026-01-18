import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
def add_revision_ids(self, revision_ids):
    """Add revision_ids to the set of revision_ids to be fetched."""
    self._explicit_rev_ids.update(revision_ids)