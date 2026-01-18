from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def read_certificate_signing_request_status(self, name, **kwargs):
    """
        read status of the specified CertificateSigningRequest
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.read_certificate_signing_request_status(name,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the CertificateSigningRequest (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :return: V1beta1CertificateSigningRequest
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.read_certificate_signing_request_status_with_http_info(name, **kwargs)
    else:
        data = self.read_certificate_signing_request_status_with_http_info(name, **kwargs)
        return data