from . import exceptions
from . import misc
from . import normalizers
def fragment_is_valid(fragment, require=False):
    """Determine if the fragment component is valid.

    :param str fragment:
        The fragment string to validate.
    :param bool require:
        (optional) Set to ``True`` to require the presence of a fragment.
    :returns:
        ``True`` if the fragment is valid. ``False`` otherwise.
    :rtype:
        bool
    """
    return is_valid(fragment, misc.FRAGMENT_MATCHER, require)