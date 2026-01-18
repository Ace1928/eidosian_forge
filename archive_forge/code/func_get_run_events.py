import collections
import os
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def get_run_events(self):
    """Returns tf.Event generators for each run's `DirectoryLoader`.

        Warning: the generators are stateful and consuming them will affect the
        results of any other existing generators for that run; calling code should
        ensure it takes events from only a single generator per run at a time.

        Returns:
          Dictionary containing an entry for each run, mapping the run name to a
          generator yielding tf.Event protobuf objects loaded from that run.
        """
    runs = list(self._directory_loaders)
    logger.info('Creating event loading generators for %d runs', len(runs))
    run_to_loader = collections.OrderedDict()
    for run_name in sorted(runs):
        loader = self._directory_loaders[run_name]
        run_to_loader[run_name] = self._wrap_loader_generator(loader.Load())
    return run_to_loader