from __future__ import absolute_import, division, print_function
@staticmethod
def format_request(method, url, *args, **kwargs):
    """
        Formats the payload from the module, into a payload the API handler can use.

        :param url: Connection URL to access
        :type url: string
        :param method: The preferred API Request method (GET, ADD, POST, etc....)
        :type method: basestring
        :param kwargs: The payload dictionary from the module to be converted.

        :return: Properly formatted dictionary payload for API Request via Connection Plugin.
        :rtype: dict
        """
    params = [{'url': url}]
    if args:
        for arg in args:
            params[0].update(arg)
    if kwargs:
        keylist = list(kwargs)
        for k in keylist:
            kwargs[k.replace('__', '-')] = kwargs.pop(k)
        if method == 'get' or method == 'clone':
            params[0].update(kwargs)
        elif kwargs.get('data', False):
            params[0]['data'] = kwargs['data']
        else:
            params[0]['data'] = kwargs
    return params