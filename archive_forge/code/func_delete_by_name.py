from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def delete_by_name(self, path, name, tenant='', tenant_uuid='', timeout=None, params=None, api_version=None, **kwargs):
    """
        Helper function to perform HTTP DELETE on Avi REST Objects using object
        type and name.Internally, it transforms the request to
        api/path?name=<name>...
        :param path: relative path to service
        :param name: name of the object
        :param tenant: overrides the tenant used during session creation
        :param tenant_uuid: overrides the tenant or tenant_uuid during session
            creation
        :param timeout: timeout for API calls; Default value is 60 seconds
        :param params: dictionary of key value pairs to be sent as query
            parameters
        :param api_version: overrides x-avi-header in request header during
            session creation
        returns session's response object
        """
    uuid = self._get_uuid_by_name(path, name, tenant, tenant_uuid, api_version=api_version)
    if not uuid:
        raise ObjectNotFound('%s/?name=%s' % (path, name))
    path = '%s/%s' % (path, uuid)
    return self.delete(path, tenant, tenant_uuid, timeout=timeout, params=params, api_version=api_version, **kwargs)