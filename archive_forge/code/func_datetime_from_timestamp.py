import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('datetime')
@specs.parameter('timestamp', yaqltypes.Number())
@specs.parameter('offset', TIMESPAN_TYPE)
def datetime_from_timestamp(timestamp, offset=ZERO_TIMESPAN):
    """:yaql:datetime

    Returns datetime object built by timestamp.

    :signature: datetime(timestamp, offset => timespan(0))
    :arg timestamp: timespan object to represent datetime
    :argType timestamp: number
    :arg offset: datetime offset in microsecond resolution, needed for tzinfo,
        timespan(0) by default
    :argType offset: timespan type
    :returnType: datetime object

    .. code::

        yaql> let(datetime(1256953732)) -> [$.year, $.month, $.day]
        [2009, 10, 31]
    """
    zone = _get_tz(offset)
    return DATETIME_TYPE.fromtimestamp(timestamp, tz=zone)