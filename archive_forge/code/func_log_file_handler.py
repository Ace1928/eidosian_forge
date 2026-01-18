from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def log_file_handler(self, logpath, **kwargs):
    """
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.log_file_handler(logpath, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str logpath: path to the log (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.log_file_handler_with_http_info(logpath, **kwargs)
    else:
        data = self.log_file_handler_with_http_info(logpath, **kwargs)
        return data