import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
def assertEventuallyTrue(self, obj, trait, condition, timeout=5.0):
    """ Assert that the given condition is eventually true.

        Parameters
        ----------
        obj : HasTraits
            The HasTraits class instance whose traits will change.

        trait : str
            The extended trait name of trait changes to listen to.

        condition : callable
            A function that will be called when the specified trait
            changes.  This should accept ``obj`` and should return a
            Boolean indicating whether the condition is satisfied or not.

        timeout : float or None, optional
            The amount of time in seconds to wait for the condition to
            become true.  None can be used to indicate no timeout.

        """
    try:
        wait_for_condition(condition=condition, obj=obj, trait=trait, timeout=timeout)
    except RuntimeError:
        condition_at_timeout = condition(obj)
        self.fail('Timed out waiting for condition. At timeout, condition was {0}.'.format(condition_at_timeout))