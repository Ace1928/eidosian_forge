from typing import ContextManager
from google.api_core.exceptions import FailedPrecondition
def require_started(self):
    if not self._started:
        raise FailedPrecondition('__enter__ has never been called.')