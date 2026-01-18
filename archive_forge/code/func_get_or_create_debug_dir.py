from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat
def get_or_create_debug_dir(export_dir):
    """Returns path to the debug sub-directory, creating if it does not exist."""
    debug_dir = get_debug_dir(export_dir)
    file_io.recursive_create_dir(debug_dir)
    return debug_dir