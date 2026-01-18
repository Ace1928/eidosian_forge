import os
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.lib.io import file_io
def remove_temp_dirpath(dirpath, strategy):
    """Removes the temp path after writing is finished.

  Args:
    dirpath: Original dirpath that would be used without distribution.
    strategy: The tf.distribute strategy object currently used.
  """
    if strategy is None:
        strategy = distribute_lib.get_strategy()
    if strategy is None:
        return
    if strategy.extended._in_multi_worker_mode() and (not strategy.extended.should_checkpoint):
        file_io.delete_recursively(_get_temp_dir(dirpath, strategy))