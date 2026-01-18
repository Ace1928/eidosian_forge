from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def iter_merge_sorted_revisions(self, start_revision_id=None, stop_revision_id=None, stop_rule='exclude', direction='reverse'):
    """Walk the revisions for a branch in merge sorted order.

        Merge sorted order is the output from a merge-aware,
        topological sort, i.e. all parents come before their
        children going forward; the opposite for reverse.

        Args:
          start_revision_id: the revision_id to begin walking from.
            If None, the branch tip is used.
          stop_revision_id: the revision_id to terminate the walk
            after. If None, the rest of history is included.
          stop_rule: if stop_revision_id is not None, the precise rule
            to use for termination:

            * 'exclude' - leave the stop revision out of the result (default)
            * 'include' - the stop revision is the last item in the result
            * 'with-merges' - include the stop revision and all of its
              merged revisions in the result
            * 'with-merges-without-common-ancestry' - filter out revisions
              that are in both ancestries
          direction: either 'reverse' or 'forward':

            * reverse means return the start_revision_id first, i.e.
              start at the most recent revision and go backwards in history
            * forward returns tuples in the opposite order to reverse.
              Note in particular that forward does *not* do any intelligent
              ordering w.r.t. depth as some clients of this API may like.
              (If required, that ought to be done at higher layers.)

        Returns: an iterator over (revision_id, depth, revno, end_of_merge)
            tuples where:

            * revision_id: the unique id of the revision
            * depth: How many levels of merging deep this node has been
              found.
            * revno_sequence: This field provides a sequence of
              revision numbers for all revisions. The format is:
              (REVNO, BRANCHNUM, BRANCHREVNO). BRANCHNUM is the number of the
              branch that the revno is on. From left to right the REVNO numbers
              are the sequence numbers within that branch of the revision.
            * end_of_merge: When True the next node (earlier in history) is
              part of a different merge.
        """
    with self.lock_read():
        if self._merge_sorted_revisions_cache is None:
            last_revision = self.last_revision()
            known_graph = self.repository.get_known_graph_ancestry([last_revision])
            self._merge_sorted_revisions_cache = known_graph.merge_sort(last_revision)
        filtered = self._filter_merge_sorted_revisions(self._merge_sorted_revisions_cache, start_revision_id, stop_revision_id, stop_rule)
        filtered = self._filter_start_non_ancestors(filtered)
        if direction == 'reverse':
            return filtered
        if direction == 'forward':
            return reversed(list(filtered))
        else:
            raise ValueError('invalid direction %r' % direction)