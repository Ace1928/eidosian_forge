from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def list_namespaced_custom_object(self, group, version, namespace, plural, **kwargs):
    """
        list or watch namespace scoped custom objects
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.list_namespaced_custom_object(group, version,
        namespace, plural, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str group: The custom resource's group name (required)
        :param str version: The custom resource's version (required)
        :param str namespace: The custom resource's namespace (required)
        :param str plural: The custom resource's plural name. For TPRs this
        would be lowercase plural kind. (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param str field_selector: A selector to restrict the list of returned
        objects by their fields. Defaults to everything.
        :param str label_selector: A selector to restrict the list of returned
        objects by their labels. Defaults to everything.
        :param str resource_version: When specified with a watch call, shows
        changes that occur after that particular version of a resource. Defaults
        to changes from the beginning of history. When specified for list: - if
        unset, then the result is returned from remote storage based on
        quorum-read flag; - if it's 0, then we simply return what we currently
        have in cache, no guarantee; - if set to non zero, then the result is at
        least as fresh as given rv.
        :param int timeout_seconds: Timeout for the list/watch call. This limits
        the duration of the call, regardless of any activity or inactivity.
        :param bool watch: Watch for changes to the described resources and
        return them as a stream of add, update, and remove notifications.
        :return: object
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.list_namespaced_custom_object_with_http_info(group, version, namespace, plural, **kwargs)
    else:
        data = self.list_namespaced_custom_object_with_http_info(group, version, namespace, plural, **kwargs)
        return data