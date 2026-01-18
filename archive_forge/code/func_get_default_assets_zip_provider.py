import os
from tensorboard.util import tb_logging
def get_default_assets_zip_provider():
    """Try to get a function to provide frontend assets.

    Returns:
      Either (a) a callable that takes no arguments and returns an open
      file handle to a Zip archive of frontend assets, or (b) `None`, if
      the frontend assets cannot be found.
    """
    path = os.path.join(os.path.dirname(__file__), 'webfiles.zip')
    if not os.path.exists(path):
        logger.warning('webfiles.zip static assets not found: %s', path)
        return None
    return lambda: open(path, 'rb')