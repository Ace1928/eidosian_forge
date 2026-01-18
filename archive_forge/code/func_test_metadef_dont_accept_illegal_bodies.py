import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def test_metadef_dont_accept_illegal_bodies(self):
    path = self._url('/v2/metadefs/namespaces/bodytest')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.NOT_FOUND, response.status_code)
    path = self._url('/v2/metadefs/namespaces')
    headers = self._headers({'content-type': 'application/json'})
    namespace_name = 'bodytest'
    data = jsonutils.dumps({'namespace': namespace_name, 'display_name': 'My User Friendly Namespace', 'description': 'My description'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    data_urls = ['/v2/schemas/metadefs/namespace', '/v2/schemas/metadefs/namespaces', '/v2/schemas/metadefs/resource_type', '/v2/schemas/metadefs/resource_types', '/v2/schemas/metadefs/property', '/v2/schemas/metadefs/properties', '/v2/schemas/metadefs/object', '/v2/schemas/metadefs/objects', '/v2/schemas/metadefs/tag', '/v2/schemas/metadefs/tags', '/v2/metadefs/resource_types']
    for value in data_urls:
        path = self._url(value)
        data = jsonutils.dumps(['body'])
        response = requests.get(path, headers=self._headers(), data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code)
    test_urls = [('/v2/metadefs/namespaces/%s/resource_types', 'get'), ('/v2/metadefs/namespaces/%s/resource_types/type', 'delete'), ('/v2/metadefs/namespaces/%s', 'get'), ('/v2/metadefs/namespaces/%s', 'delete'), ('/v2/metadefs/namespaces/%s/objects/name', 'get'), ('/v2/metadefs/namespaces/%s/objects/name', 'delete'), ('/v2/metadefs/namespaces/%s/properties', 'get'), ('/v2/metadefs/namespaces/%s/tags/test', 'get'), ('/v2/metadefs/namespaces/%s/tags/test', 'post'), ('/v2/metadefs/namespaces/%s/tags/test', 'delete')]
    for link, method in test_urls:
        path = self._url(link % namespace_name)
        data = jsonutils.dumps(['body'])
        response = getattr(requests, method)(path, headers=self._headers(), data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code)