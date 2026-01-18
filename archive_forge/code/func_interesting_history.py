import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def interesting_history(self):
    if self.revision:
        rev1, rev2 = builtins._get_revision_range(self.revision, self.branch, 'fast-export')
        start_rev_id = rev1.rev_id
        end_rev_id = rev2.rev_id
    else:
        start_rev_id = None
        end_rev_id = None
    self.note('Calculating the revisions to include ...')
    view_revisions = [rev_id for rev_id, _, _, _ in self.branch.iter_merge_sorted_revisions(end_rev_id, start_rev_id)]
    view_revisions.reverse()
    if start_rev_id is not None:
        self.note('Calculating the revisions to exclude ...')
        self.excluded_revisions = {rev_id for rev_id, _, _, _ in self.branch.iter_merge_sorted_revisions(start_rev_id)}
        if self.baseline:
            self.excluded_revisions.remove(start_rev_id)
            view_revisions.insert(0, start_rev_id)
    return list(view_revisions)