import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def test_metadef_resource_types_lifecycle(self):
    path = self._url('/v2/metadefs/namespaces/MyNamespace')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.NOT_FOUND, response.status_code)
    path = self._url('/v2/metadefs/namespaces')
    headers = self._headers({'content-type': 'application/json'})
    namespace_name = 'MyNamespace'
    data = jsonutils.dumps({'namespace': namespace_name, 'display_name': 'My User Friendly Namespace', 'description': 'My description', 'visibility': 'public', 'protected': False, 'owner': 'The Test Owner'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    path = self._url('/v2/metadefs/namespaces/%s/resource_types' % namespace_name)
    response = requests.get(path, headers=self._headers())
    metadef_resource_type = jsonutils.loads(response.text)
    self.assertEqual(0, len(metadef_resource_type['resource_type_associations']))
    path = self._url('/v2/metadefs/namespaces/MyNamespace/resource_types')
    headers = self._headers({'content-type': 'application/json'})
    metadef_resource_type_name = 'resource_type1'
    data = jsonutils.dumps({'name': 'resource_type1', 'prefix': 'hw_', 'properties_target': 'image'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CONFLICT, response.status_code)
    path = self._url('/v2/metadefs/namespaces/%s/resource_types' % namespace_name)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    metadef_resource_type = jsonutils.loads(response.text)
    self.assertEqual('resource_type1', metadef_resource_type['resource_type_associations'][0]['name'])
    resource_type = jsonutils.loads(response.text)
    checked_keys = set([u'name', u'prefix', u'properties_target', u'created_at', u'updated_at'])
    self.assertEqual(set(resource_type['resource_type_associations'][0].keys()), checked_keys)
    expected_metadef_resource_types = {'name': metadef_resource_type_name, 'prefix': 'hw_', 'properties_target': 'image'}
    checked_values = set([u'name', u'prefix', u'properties_target'])
    for key, value in expected_metadef_resource_types.items():
        if key in checked_values:
            self.assertEqual(resource_type['resource_type_associations'][0][key], value, key)
    path = self._url('/v2/metadefs/namespaces/%s/resource_types/%s' % (namespace_name, metadef_resource_type_name))
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/metadefs/namespaces/%s/resource_types' % namespace_name)
    response = requests.get(path, headers=self._headers())
    metadef_resource_type = jsonutils.loads(response.text)
    self.assertEqual(0, len(metadef_resource_type['resource_type_associations']))