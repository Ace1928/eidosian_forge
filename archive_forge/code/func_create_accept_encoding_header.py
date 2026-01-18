from collections import namedtuple
import re
import textwrap
import warnings
def create_accept_encoding_header(header_value):
    """
    Create an object representing the ``Accept-Encoding`` header in a request.

    :param header_value: (``str``) header value
    :return: If `header_value` is ``None``, an :class:`AcceptEncodingNoHeader`
             instance.

             | If `header_value` is a valid ``Accept-Encoding`` header, an
               :class:`AcceptEncodingValidHeader` instance.

             | If `header_value` is an invalid ``Accept-Encoding`` header, an
               :class:`AcceptEncodingInvalidHeader` instance.
    """
    if header_value is None:
        return AcceptEncodingNoHeader()
    if isinstance(header_value, AcceptEncoding):
        return header_value.copy()
    try:
        return AcceptEncodingValidHeader(header_value=header_value)
    except ValueError:
        return AcceptEncodingInvalidHeader(header_value=header_value)