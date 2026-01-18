from . import exceptions
from . import misc
from . import normalizers
def authority_is_valid(authority, host=None, require=False):
    """Determine if the authority string is valid.

    :param str authority:
        The authority to validate.
    :param str host:
        (optional) The host portion of the authority to validate.
    :param bool require:
        (optional) Specify if authority must not be None.
    :returns:
        ``True`` if valid, ``False`` otherwise
    :rtype:
        bool
    """
    validated = is_valid(authority, misc.SUBAUTHORITY_MATCHER, require)
    if validated and host is not None:
        return host_is_valid(host, require)
    return validated