import collections
import os
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def synchronize_runs(self):
    """Finds new runs within `logdir` and makes `DirectoryLoaders` for
        them.

        In addition, any existing `DirectoryLoader` whose run directory
        no longer exists will be deleted.
        """
    logger.info('Starting logdir traversal of %s', self._logdir)
    runs_seen = set()
    for subdir in io_wrapper.GetLogdirSubdirectories(self._logdir):
        run = os.path.relpath(subdir, self._logdir)
        runs_seen.add(run)
        if run not in self._directory_loaders:
            logger.info('- Adding run for relative directory %s', run)
            self._directory_loaders[run] = self._directory_loader_factory(subdir)
    stale_runs = set(self._directory_loaders) - runs_seen
    if stale_runs:
        for run in stale_runs:
            logger.info('- Removing run for relative directory %s', run)
            del self._directory_loaders[run]
    logger.info('Ending logdir traversal of %s', self._logdir)