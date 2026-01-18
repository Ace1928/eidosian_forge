from collections import namedtuple
import re
import textwrap
import warnings
def create_accept_header(header_value):
    """
    Create an object representing the ``Accept`` header in a request.

    :param header_value: (``str``) header value
    :return: If `header_value` is ``None``, an :class:`AcceptNoHeader`
             instance.

             | If `header_value` is a valid ``Accept`` header, an
               :class:`AcceptValidHeader` instance.

             | If `header_value` is an invalid ``Accept`` header, an
               :class:`AcceptInvalidHeader` instance.
    """
    if header_value is None:
        return AcceptNoHeader()
    if isinstance(header_value, Accept):
        return header_value.copy()
    try:
        return AcceptValidHeader(header_value=header_value)
    except ValueError:
        return AcceptInvalidHeader(header_value=header_value)