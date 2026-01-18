from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat
def get_or_create_variables_dir(export_dir):
    """Return variables sub-directory, or create one if it doesn't exist."""
    variables_dir = get_variables_dir(export_dir)
    file_io.recursive_create_dir(variables_dir)
    return variables_dir