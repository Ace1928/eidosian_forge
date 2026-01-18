import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
def is_timespan(value):
    """:yaql:isTimespan
    Returns true if value is timespan object, false otherwise.

    :signature: isTimespan(value)
    :arg value: input value
    :argType value: any
    :returnType: boolean

    .. code::

        yaql> isTimespan(now())
        false
        yaql> isTimespan(timespan())
        true
    """
    return isinstance(value, TIMESPAN_TYPE)