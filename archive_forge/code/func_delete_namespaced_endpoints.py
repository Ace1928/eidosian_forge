from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def delete_namespaced_endpoints(self, name, namespace, **kwargs):
    """
        delete Endpoints
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.delete_namespaced_endpoints(name, namespace,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the Endpoints (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param V1DeleteOptions body:
        :param str dry_run: When present, indicates that modifications should
        not be persisted. An invalid or unrecognized dryRun directive will
        result in an error response and no further processing of the request.
        Valid values are: - All: all dry run stages will be processed
        :param int grace_period_seconds: The duration in seconds before the
        object should be deleted. Value must be non-negative integer. The value
        zero indicates delete immediately. If this value is nil, the default
        grace period for the specified type will be used. Defaults to a per
        object value if not specified. zero means delete immediately.
        :param bool orphan_dependents: Deprecated: please use the
        PropagationPolicy, this field will be deprecated in 1.7. Should the
        dependent objects be orphaned. If true/false, the "orphan" finalizer
        will be added to/removed from the object's finalizers list. Either this
        field or PropagationPolicy may be set, but not both.
        :param str propagation_policy: Whether and how garbage collection will
        be performed. Either this field or OrphanDependents may be set, but not
        both. The default policy is decided by the existing finalizer set in the
        metadata.finalizers and the resource-specific default policy. Acceptable
        values are: 'Orphan' - orphan the dependents; 'Background' - allow the
        garbage collector to delete the dependents in the background;
        'Foreground' - a cascading policy that deletes all dependents in the
        foreground.
        :return: V1Status
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.delete_namespaced_endpoints_with_http_info(name, namespace, **kwargs)
    else:
        data = self.delete_namespaced_endpoints_with_http_info(name, namespace, **kwargs)
        return data