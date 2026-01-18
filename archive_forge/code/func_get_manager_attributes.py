from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.common.text.converters import to_native
def get_manager_attributes(self):
    result = {}
    manager_attributes = []
    properties = ['Attributes', 'Id']
    response = self.get_request(self.root_uri + self.manager_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    try:
        for members in data[u'Links'][u'Oem'][u'Dell'][u'DellAttributes']:
            attributes_uri = members[u'@odata.id']
            response = self.get_request(self.root_uri + attributes_uri)
            if response['ret'] is False:
                return response
            data = response['data']
            attributes = {}
            for prop in properties:
                if prop in data:
                    attributes[prop] = data.get(prop)
            if attributes:
                manager_attributes.append(attributes)
        result['ret'] = True
    except (AttributeError, KeyError) as e:
        result['ret'] = False
        result['msg'] = 'Failed to find attribute/key: ' + str(e)
    result['entries'] = manager_attributes
    return result