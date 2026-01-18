import warnings
from warnings import warn
import breezy
def suppress_deprecation_warnings(override=True):
    """Call this function to suppress all deprecation warnings.

    When this is a final release version, we don't want to annoy users with
    lots of deprecation warnings. We only want the deprecation warnings when
    running a dev or release candidate.

    :param override: If True, always set the ignore, if False, only set the
        ignore if there isn't already a filter.

    :return: A callable to remove the new warnings this added.
    """
    if not override and _check_for_filter(error_only=False):
        filter = None
    else:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        filter = warnings.filters[0]
    return _remove_filter_callable(filter)