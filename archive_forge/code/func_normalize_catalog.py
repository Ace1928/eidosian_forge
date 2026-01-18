import abc
import copy
from keystoneauth1 import discover
from keystoneauth1 import exceptions
def normalize_catalog(self):
    """Return the catalog normalized into v3 format."""
    catalog = []
    for service in copy.deepcopy(self._catalog):
        if 'type' not in service:
            continue
        service.setdefault('name', None)
        service.setdefault('id', None)
        service['endpoints'] = self._normalize_endpoints(service.get('endpoints', []))
        for endpoint in service['endpoints']:
            endpoint['region_name'] = self._get_endpoint_region(endpoint)
            endpoint.setdefault('id', None)
        catalog.append(service)
    return catalog