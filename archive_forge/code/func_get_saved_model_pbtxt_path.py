from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat
def get_saved_model_pbtxt_path(export_dir):
    return file_io.join(compat.as_bytes(compat.path_to_str(export_dir)), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))