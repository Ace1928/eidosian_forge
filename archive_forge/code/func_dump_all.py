import collections
from requests import compat
def dump_all(response, request_prefix=b'< ', response_prefix=b'> '):
    """Dump all requests and responses including redirects.

    This takes the response returned by requests and will dump all
    request-response pairs in the redirect history in order followed by the
    final request-response.

    Example::

        import requests
        from requests_toolbelt.utils import dump

        resp = requests.get('https://httpbin.org/redirect/5')
        data = dump.dump_all(resp)
        print(data.decode('utf-8'))

    :param response:
        The response to format
    :type response: :class:`requests.Response`
    :param request_prefix: (*optional*)
        Bytes to prefix each line of the request data
    :type request_prefix: :class:`bytes`
    :param response_prefix: (*optional*)
        Bytes to prefix each line of the response data
    :type response_prefix: :class:`bytes`
    :returns: Formatted bytes of request and response information.
    :rtype: :class:`bytearray`
    """
    data = bytearray()
    history = list(response.history[:])
    history.append(response)
    for response in history:
        dump_response(response, request_prefix, response_prefix, data)
    return data