import warnings
from warnings import warn
import breezy
def deprecation_string(a_callable, deprecation_version):
    """Generate an automatic deprecation string for a_callable.

    :param a_callable: The callable to substitute into deprecation_version.
    :param deprecation_version: A deprecation format warning string. This
        should have a single %s operator in it. a_callable will be turned into
        a nice python symbol and then substituted into deprecation_version.
    """
    if getattr(a_callable, '__self__', None) is not None:
        symbol = '{}.{}.{}'.format(a_callable.__self__.__class__.__module__, a_callable.__self__.__class__.__name__, a_callable.__name__)
    elif getattr(a_callable, '__qualname__', None) is not None and '<' not in a_callable.__qualname__:
        symbol = '{}.{}'.format(a_callable.__module__, a_callable.__qualname__)
    else:
        symbol = '{}.{}'.format(a_callable.__module__, a_callable.__name__)
    return deprecation_version % symbol