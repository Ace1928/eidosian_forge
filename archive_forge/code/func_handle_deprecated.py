import logging
import warnings
from keystoneauth1 import identity
from keystoneauth1 import session as k_session
from monascaclient.osc import migration
from monascaclient import version
def handle_deprecated(args, kwargs):
    """Handles all deprecations.

    Method goes through passed args and kwargs
    and handles all values that are invalid from POV
    of current client but:

    * has their counterparts
    * are candidates to be dropped

    """
    kwargs.update(_handle_deprecated_args(args))
    _handle_deprecated_kwargs(kwargs)