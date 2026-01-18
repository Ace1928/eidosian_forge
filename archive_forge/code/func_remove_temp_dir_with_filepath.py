import os
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.lib.io import file_io
def remove_temp_dir_with_filepath(filepath, strategy):
    """Removes the temp path for file after writing is finished.

  Args:
    filepath: Original filepath that would be used without distribution.
    strategy: The tf.distribute strategy object currently used.
  """
    remove_temp_dirpath(os.path.dirname(filepath), strategy)