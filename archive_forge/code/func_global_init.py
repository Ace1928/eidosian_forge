import os
import sys
import absl.logging
from tensorboard.compat import tf
def global_init():
    """Modifies the global environment for running TensorBoard as main.

    This functions changes global state in the Python process, so it should
    not be called from library routines.
    """
    os.environ['GCS_READ_CACHE_DISABLED'] = '1'
    if getattr(tf, '__version__', 'stub') == 'stub':
        print('TensorFlow installation not found - running with reduced feature set.', file=sys.stderr)
    absl.logging.set_verbosity(absl.logging.WARNING)