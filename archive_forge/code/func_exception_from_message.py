import time
from email.utils import mktime_tz, parsedate_tz
def exception_from_message(code, message, headers=None):
    """
    Return an instance of BaseHTTPException or subclass based on response code.

    If headers include Retry-After, RFC 2616 says that its value may be one of
    two formats: HTTP-date or delta-seconds, for example:

    Retry-After: Fri, 31 Dec 1999 23:59:59 GMT
    Retry-After: 120

    If Retry-After comes in HTTP-date, it'll be translated to a positive
    delta-seconds value when passing it to the exception constructor.

    Also, RFC 2616 says that Retry-After isn't just only applicable to 429
    HTTP status, but also to other responses, like 503 and 3xx.

    Usage::
        raise exception_from_message(code=self.status,
                                     message=self.parse_error(),
                                     headers=self.headers)
    """
    kwargs = {'code': code, 'message': message, 'headers': headers}
    if headers and 'retry-after' in headers:
        http_date = parsedate_tz(headers['retry-after'])
        if http_date is not None:
            delay = max(0, int(mktime_tz(http_date) - time.time()))
            headers['retry-after'] = str(delay)
    cls = _code_map.get(code, BaseHTTPError)
    return cls(**kwargs)