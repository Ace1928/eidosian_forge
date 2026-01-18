from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat
def get_variables_dir(export_dir):
    """Return variables sub-directory in the SavedModel."""
    return file_io.join(compat.as_text(export_dir), compat.as_text(constants.VARIABLES_DIRECTORY))