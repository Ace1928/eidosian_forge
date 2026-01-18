import calendar
from datetime import (
from email.utils import (
import time
from webob.compat import (
def parse_date_delta(value):
    """
    like parse_date, but also handle delta seconds
    """
    if not value:
        return None
    try:
        value = int(value)
    except ValueError:
        return parse_date(value)
    else:
        return _now() + timedelta(seconds=value)