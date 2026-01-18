from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
def log_import_warning(self, logger='pyomo', msg=None):
    """Log the import error message to the specified logger

        This will log the the import error message to the specified
        logger.  If ``msg=`` is specified, it will override the default
        message passed to this instance of
        :py:class:`ModuleUnavailable`.

        """
    logging.getLogger(logger).warning(self._moduleunavailable_message(msg))