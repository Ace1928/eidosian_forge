from . import exceptions
from . import misc
from . import normalizers
def path_is_valid(path, require=False):
    """Determine if the path component is valid.

    :param str path:
        The path string to validate.
    :param bool require:
        (optional) Set to ``True`` to require the presence of a path.
    :returns:
        ``True`` if the path is valid. ``False`` otherwise.
    :rtype:
        bool
    """
    return is_valid(path, misc.PATH_MATCHER, require)