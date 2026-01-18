import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('#operator_+')
@specs.parameter('dt', yaqltypes.DateTime())
@specs.parameter('ts', TIMESPAN_TYPE)
def datetime_plus_timespan(dt, ts):
    """:yaql:operator +

    Returns datetime object with added timespan.

    :signature: left + right
    :arg left: input datetime object
    :argType left: datetime object
    :arg right: input timespan object
    :argType right: timespan object
    :returnType: datetime object

    .. code::

        yaql> let(now() + timespan(days => 100)) -> $.month
        10
    """
    return dt + ts