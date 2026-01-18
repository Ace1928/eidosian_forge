import json
import os
from collections import defaultdict
import hashlib
import tempfile
from functools import partial
import kubernetes.dynamic
import kubernetes.dynamic.discovery
from kubernetes import __version__
from kubernetes.dynamic.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.client.resource import (
def get_resources_for_api_version(self, prefix, group, version, preferred):
    """returns a dictionary of resources associated with provided (prefix, group, version)"""
    resources = defaultdict(list)
    subresources = defaultdict(dict)
    path = '/'.join(filter(None, [prefix, group, version]))
    try:
        resources_response = self.client.request('GET', path).resources or []
    except ServiceUnavailableError:
        resources_response = []
    resources_raw = list(filter(lambda resource: '/' not in resource['name'], resources_response))
    subresources_raw = list(filter(lambda resource: '/' in resource['name'], resources_response))
    for subresource in subresources_raw:
        resource, name = subresource['name'].split('/', 1)
        subresources[resource][name] = subresource
    for resource in resources_raw:
        for key in ('prefix', 'group', 'api_version', 'client', 'preferred'):
            resource.pop(key, None)
        resourceobj = kubernetes.dynamic.Resource(prefix=prefix, group=group, api_version=version, client=self.client, preferred=preferred, subresources=subresources.get(resource['name']), **resource)
        resources[resource['kind']].append(resourceobj)
        resource_lookup = {'prefix': prefix, 'group': group, 'api_version': version, 'kind': resourceobj.kind, 'name': resourceobj.name}
        resource_list = ResourceList(self.client, group=group, api_version=version, base_kind=resource['kind'], base_resource_lookup=resource_lookup)
        resources[resource_list.kind].append(resource_list)
    return resources