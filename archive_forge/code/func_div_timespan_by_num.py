import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('#operator_/')
@specs.parameter('ts', TIMESPAN_TYPE)
@specs.parameter('n', yaqltypes.Number())
def div_timespan_by_num(ts, n):
    """:yaql:operator /

    Returns timespan object divided by number.

    :signature: left / right
    :arg left: left timespan object
    :argType left: timespan object
    :arg right: number to divide by
    :argType right: number
    :returnType: timespan object

    .. code::

        yaql> let(timespan(hours => 24) / 2) -> $.hours
        12.0
    """
    return TIMESPAN_TYPE(microseconds=microseconds(ts) / n)