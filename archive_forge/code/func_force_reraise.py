import functools
import io
import logging
import os
import sys
import time
import traceback
from oslo_utils import encodeutils
from oslo_utils import reflection
from oslo_utils import timeutils
def force_reraise(self):
    if self.type_ is None and self.value is None:
        raise RuntimeError('There is no (currently) captured exception to force the reraising of')
    try:
        if self.value is None:
            self.value = self.type_()
        if self.value.__traceback__ is not self.tb:
            raise self.value.with_traceback(self.tb)
        raise self.value
    finally:
        self.value = None
        self.tb = None