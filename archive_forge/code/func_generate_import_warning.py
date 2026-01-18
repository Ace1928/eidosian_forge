from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
@deprecated('use :py:class:`log_import_warning()`', version='6.0')
def generate_import_warning(self, logger='pyomo.common'):
    self.log_import_warning(logger)