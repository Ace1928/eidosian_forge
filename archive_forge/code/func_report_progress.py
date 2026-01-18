import time
import configobj
from fastimport import commands
from fastimport import errors as plugin_errors
from fastimport import processor
from fastimport.helpers import invert_dictset
from .... import debug, delta, errors, osutils, progress
from .... import revision as _mod_revision
from ....bzr.knitpack_repo import KnitPackRepository
from ....trace import mutter, note, warning
from .. import (branch_updater, cache_manager, helpers, idmapfile, marks_file,
def report_progress(self, details=''):
    if self._revision_count % self.progress_every == 0:
        if self.total_commits is not None:
            counts = '%d/%d' % (self._revision_count, self.total_commits)
        else:
            counts = '%d' % (self._revision_count,)
        minutes = (time.time() - self._start_time) / 60
        revisions_added = self._revision_count - self.skip_total
        rate = revisions_added * 1.0 / minutes
        if rate > 10:
            rate_str = 'at %.0f/minute ' % rate
        else:
            rate_str = 'at %.1f/minute ' % rate
        self.note('%s commits processed %s%s' % (counts, rate_str, details))