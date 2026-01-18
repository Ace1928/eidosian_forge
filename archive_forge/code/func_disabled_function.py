imported modules too. Instead, we just create new functions with
from __future__ import division, absolute_import, print_function
from future import utils
def disabled_function(name):
    """
    Returns a function that cannot be called
    """

    def disabled(*args, **kwargs):
        """
        A function disabled by the ``future`` module. This function is
        no longer a builtin in Python 3.
        """
        raise NameError('obsolete Python 2 builtin {0} is disabled'.format(name))
    return disabled