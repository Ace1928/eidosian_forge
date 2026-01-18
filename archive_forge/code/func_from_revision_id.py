from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
@staticmethod
def from_revision_id(branch, revision_id):
    """Construct a RevisionInfo given just the id.

        Use this if you don't know or care what the revno is.
        """
    return RevisionInfo(branch, revno=None, rev_id=revision_id)