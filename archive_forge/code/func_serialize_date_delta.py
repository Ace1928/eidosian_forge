import calendar
from datetime import (
from email.utils import (
import time
from webob.compat import (
def serialize_date_delta(value):
    if isinstance(value, (float, int, long)):
        return str(int(value))
    else:
        return serialize_date(value)