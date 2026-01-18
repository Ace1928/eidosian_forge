import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('#operator_>')
@specs.parameter('dt1', yaqltypes.DateTime())
@specs.parameter('dt2', yaqltypes.DateTime())
def datetime_gt_datetime(dt1, dt2):
    """:yaql:operator >

    Returns true if left datetime is strictly greater than right datetime,
    false otherwise.

    :signature: left > right
    :arg left: left datetime object
    :argType left: datetime object
    :arg right: right datetime object
    :argType right: datetime object
    :returnType: boolean

    .. code::

        yaql> datetime(2011, 11, 11) > datetime(2010, 10, 10)
        true
    """
    return dt1 > dt2