from __future__ import absolute_import
import os
import re
import json
import mimetypes
import tempfile
from multiprocessing.pool import ThreadPool
from datetime import date, datetime
from six import PY3, integer_types, iteritems, text_type
from six.moves.urllib.parse import quote
from . import models
from .configuration import Configuration
from .rest import ApiException, RESTClientObject
def call_api(self, resource_path, method, path_params=None, query_params=None, header_params=None, body=None, post_params=None, files=None, response_type=None, auth_settings=None, async_req=None, _return_http_data_only=None, collection_formats=None, _preload_content=True, _request_timeout=None):
    """
        Makes the HTTP request (synchronous) and return the deserialized data.
        To make an async request, set the async parameter.

        :param resource_path: Path to method endpoint.
        :param method: Method to call.
        :param path_params: Path parameters in the url.
        :param query_params: Query parameters in the url.
        :param header_params: Header parameters to be
            placed in the request header.
        :param body: Request body.
        :param post_params dict: Request post form parameters,
            for `application/x-www-form-urlencoded`, `multipart/form-data`.
        :param auth_settings list: Auth Settings names for the request.
        :param response: Response data type.
        :param files dict: key -> filename, value -> filepath,
            for `multipart/form-data`.
        :param async_req bool: execute request asynchronously
        :param _return_http_data_only: response data without head status code
        and headers
        :param collection_formats: dict of collection formats for path, query,
            header, and post parameters.
        :param _preload_content: if False, the urllib3.HTTPResponse object will
        be returned without
                                 reading/decoding response data. Default is
                                 True.
        :param _request_timeout: timeout setting for this request. If one number
        provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return:
            If async parameter is True,
            the request will be called asynchronously.
            The method will return the request thread.
            If parameter async is False or missing,
            then the method will return the response directly.
        """
    if not async_req:
        return self.__call_api(resource_path, method, path_params, query_params, header_params, body, post_params, files, response_type, auth_settings, _return_http_data_only, collection_formats, _preload_content, _request_timeout)
    else:
        thread = self.pool.apply_async(self.__call_api, (resource_path, method, path_params, query_params, header_params, body, post_params, files, response_type, auth_settings, _return_http_data_only, collection_formats, _preload_content, _request_timeout))
    return thread