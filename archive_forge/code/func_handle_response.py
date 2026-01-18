import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
def handle_response(url, code, getheader, data):
    """Interpret the code & headers and wrap the provided data in a RangeFile.

    This is a factory method which returns an appropriate RangeFile based on
    the code & headers it's given.

    :param url: The url being processed. Mostly for error reporting
    :param code: The integer HTTP response code
    :param getheader: Function for retrieving header
    :param data: A file-like object that can be read() to get the
                 requested data
    :return: A file-like object that can seek()+read() the
             ranges indicated by the headers.
    """
    if code == 200:
        rfile = ResponseFile(url, data)
    elif code == 206:
        rfile = RangeFile(url, data)
        content_type = getheader('content-type', 'application/octet-stream')
        mimetype, options = cgi.parse_header(content_type)
        if mimetype == 'multipart/byteranges':
            rfile.set_boundary(options['boundary'].encode('ascii'))
        else:
            content_range = getheader('content-range', None)
            if content_range is None:
                raise errors.InvalidHttpResponse(url, 'Missing the Content-Range header in a 206 range response')
            rfile.set_range_from_header(content_range)
    else:
        raise errors.UnexpectedHttpStatus(url, code)
    return rfile