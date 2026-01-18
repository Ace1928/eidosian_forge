from urllib import parse
def safe_urlencode(params_dict):
    """Workaround incompatible change to urllib.parse

    urllib's parse library used to adhere to RFC 2396 until
    python 3.7. The library moved from RFC 2396 to RFC 3986
    for quoting URL strings in python 3.7 and '~' is now
    included in the set of reserved characters. [1]

    This utility ensures "~" is never encoded.

    See LP 1785283 [2] for more details.
    [1] https://docs.python.org/3/library/urllib.parse.html#url-quoting
    [2] https://bugs.launchpad.net/python-manilaclient/+bug/1785283

    :param params_dict can be a list of (k,v) tuples, or a dictionary
    """
    parsed_params = parse.urlencode(params_dict)
    return parsed_params.replace('%7E', '~')