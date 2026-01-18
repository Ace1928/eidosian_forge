from blazarclient import base
from blazarclient import exception
from blazarclient.i18n import _
def list_properties(self, detail=False, all=False, sort_by=None):
    url = '/os-hosts/properties'
    query_parts = []
    if detail:
        query_parts.append('detail=True')
    if all:
        query_parts.append('all=True')
    if query_parts:
        url += '?' + '&'.join(query_parts)
    resp, body = self.request_manager.get(url)
    resource_properties = body['resource_properties']
    if detail:
        for p in resource_properties:
            p['property_values'] = p['values']
            del p['values']
    if sort_by:
        resource_properties = sorted(resource_properties, key=lambda l: l[sort_by])
    return resource_properties