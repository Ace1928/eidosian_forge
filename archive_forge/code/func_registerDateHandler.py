from .asctime import _parse_date_asctime
from .greek import _parse_date_greek
from .hungarian import _parse_date_hungarian
from .iso8601 import _parse_date_iso8601
from .korean import _parse_date_onblog, _parse_date_nate
from .perforce import _parse_date_perforce
from .rfc822 import _parse_date_rfc822
from .w3dtf import _parse_date_w3dtf
def registerDateHandler(func):
    """Register a date handler function (takes string, returns 9-tuple date in GMT)"""
    _date_handlers.insert(0, func)