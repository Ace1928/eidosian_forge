import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('#unary_operator_-')
@specs.parameter('ts', TIMESPAN_TYPE)
def negative_timespan(ts):
    """:yaql:operator unary -

    Returns negative timespan.

    :signature: -arg
    :arg arg: input timespan object
    :argType arg: timespan object
    :returnType: timespan object

    .. code::

        yaql> let(-timespan(hours => 24)) -> $.hours
        -24.0
    """
    return -ts