from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat
def get_or_create_assets_dir(export_dir):
    """Return assets sub-directory, or create one if it doesn't exist."""
    assets_destination_dir = get_assets_dir(export_dir)
    file_io.recursive_create_dir(assets_destination_dir)
    return assets_destination_dir