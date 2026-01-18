import argparse
import warnings
from http.cookies import SimpleCookie
from shlex import split
from urllib.parse import urlparse
from w3lib.http import basic_auth_header
def curl_to_request_kwargs(curl_command: str, ignore_unknown_options: bool=True) -> dict:
    """Convert a cURL command syntax to Request kwargs.

    :param str curl_command: string containing the curl command
    :param bool ignore_unknown_options: If true, only a warning is emitted when
                                        cURL options are unknown. Otherwise
                                        raises an error. (default: True)
    :return: dictionary of Request kwargs
    """
    curl_args = split(curl_command)
    if curl_args[0] != 'curl':
        raise ValueError('A curl command must start with "curl"')
    parsed_args, argv = curl_parser.parse_known_args(curl_args[1:])
    if argv:
        msg = f'Unrecognized options: {', '.join(argv)}'
        if ignore_unknown_options:
            warnings.warn(msg)
        else:
            raise ValueError(msg)
    url = parsed_args.url
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = 'http://' + url
    method = parsed_args.method or 'GET'
    result = {'method': method.upper(), 'url': url}
    headers, cookies = _parse_headers_and_cookies(parsed_args)
    if headers:
        result['headers'] = headers
    if cookies:
        result['cookies'] = cookies
    if parsed_args.data:
        result['body'] = parsed_args.data
        if not parsed_args.method:
            result['method'] = 'POST'
    return result