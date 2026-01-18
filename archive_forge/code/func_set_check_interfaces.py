import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
@contextlib.contextmanager
def set_check_interfaces(self, check_interfaces_value):
    """
        Context manager to temporarily set has_traits.CHECK_INTERFACES
        to the given value.

        Parameters
        ----------
        check_interfaces_value : int
            One of 0 (don't check), 1 (check and log a warning on interface
            mismatch) or 2 (check and raise on interface mismatch).

        Returns
        -------
        context manager
        """
    old_check_interfaces = has_traits.CHECK_INTERFACES
    has_traits.CHECK_INTERFACES = check_interfaces_value
    try:
        yield
    finally:
        has_traits.CHECK_INTERFACES = old_check_interfaces