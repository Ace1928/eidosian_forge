import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def error_from(exc, cause=None):
    """
    backward compat hack to suppress exception cause in python3.3+

    one python < 3.3 support is dropped, can replace all uses with "raise exc from None"
    """
    exc.__cause__ = cause
    exc.__suppress_context__ = True
    return exc