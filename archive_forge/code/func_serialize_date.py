import calendar
from datetime import (
from email.utils import (
import time
from webob.compat import (
def serialize_date(dt):
    if isinstance(dt, (bytes, text_type)):
        return native_(dt)
    if isinstance(dt, timedelta):
        dt = _now() + dt
    if isinstance(dt, (datetime, date)):
        dt = dt.timetuple()
    if isinstance(dt, (tuple, time.struct_time)):
        dt = calendar.timegm(dt)
    if not (isinstance(dt, float) or isinstance(dt, integer_types)):
        raise ValueError('You must pass in a datetime, date, time tuple, or integer object, not %r' % dt)
    return formatdate(dt, usegmt=True)