from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat
def get_variables_path(export_dir):
    """Return the variables path, used as the prefix for checkpoint files."""
    return file_io.join(compat.as_text(get_variables_dir(export_dir)), compat.as_text(constants.VARIABLES_FILENAME))