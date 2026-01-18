import collections
import copy
import io
import os
import sys
import traceback
from oslo_utils import encodeutils
from oslo_utils import reflection
from taskflow import exceptions as exc
from taskflow.utils import iter_utils
from taskflow.utils import schema_utils as su
@property
def exception_str(self):
    """String representation of exception."""
    return self._exception_str