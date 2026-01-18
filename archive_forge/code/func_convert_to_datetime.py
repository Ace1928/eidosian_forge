from .core import *
from .helpers import DelimitedList, any_open_tag, any_close_tag
from datetime import datetime
@staticmethod
def convert_to_datetime(fmt: str='%Y-%m-%dT%H:%M:%S.%f'):
    """Helper to create a parse action for converting parsed
        datetime string to Python datetime.datetime

        Params -
        - fmt - format to be passed to datetime.strptime (default= ``"%Y-%m-%dT%H:%M:%S.%f"``)

        Example::

            dt_expr = pyparsing_common.iso8601_datetime.copy()
            dt_expr.set_parse_action(pyparsing_common.convert_to_datetime())
            print(dt_expr.parse_string("1999-12-31T23:59:59.999"))

        prints::

            [datetime.datetime(1999, 12, 31, 23, 59, 59, 999000)]
        """

    def cvt_fn(s, l, t):
        try:
            return datetime.strptime(t[0], fmt)
        except ValueError as ve:
            raise ParseException(s, l, str(ve))
    return cvt_fn