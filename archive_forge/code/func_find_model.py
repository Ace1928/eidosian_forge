from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import quote
import json
import re
import xml.etree.ElementTree as ET
def find_model(self, search_criteria, ret_attrs=None):
    """
        Search for a model in /models
        :param search_criteria: The XML <rs:search-criteria>
        :type search_criteria: str
        :param ret_attrs: List of attributes by name or ID to return back
            (default is Model_Handle)
        :type ret_attrs: list
        returns: Dictionary mapping of ret_attrs to values: {ret_attr: ret_val}
        rtype: dict
        """
    if ret_attrs is None:
        ret_attrs = ['Model_Handle']
    rqstd_attrs = ''
    for ra in ret_attrs:
        _id = self.attr_id(ra) or ra
        rqstd_attrs += '<rs:requested-attribute id="%s" />' % (self.attr_id(ra) or ra)
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<rs:model-request throttlesize="5"\nxmlns:rs="http://www.ca.com/spectrum/restful/schema/request"\nxmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\nxsi:schemaLocation="http://www.ca.com/spectrum/restful/schema/request ../../../xsd/Request.xsd">\n    <rs:target-models>\n        <rs:models-search>\n            <rs:search-criteria xmlns="http://www.ca.com/spectrum/restful/schema/filter">\n                {0}\n            </rs:search-criteria>\n        </rs:models-search>\n    </rs:target-models>\n {1}\n </rs:model-request>\n'.format(search_criteria, rqstd_attrs)
    url = self.build_url('/models')
    resp, info = fetch_url(self.module, url, data=xml, method='POST', use_proxy=self.module.params['use_proxy'], headers={'Content-Type': 'application/xml', 'Accept': 'application/xml'})
    status_code = info['status']
    if status_code >= 400:
        body = info['body']
    else:
        body = '' if resp is None else resp.read()
    if status_code != 200:
        self.result['msg'] = 'HTTP POST error %s: %s: %s' % (status_code, url, body)
        self.module.fail_json(**self.result)
    root = ET.fromstring(body)
    total_models = int(root.attrib['total-models'])
    error = root.attrib['error']
    model_responses = root.find('ca:model-responses', self.resp_namespace)
    if total_models < 1:
        self.result['msg'] = "No models found matching search criteria `%s'" % search_criteria
        self.module.fail_json(**self.result)
    elif total_models > 1:
        self.result['msg'] = "More than one model found (%s): `%s'" % (total_models, ET.tostring(model_responses, encoding='unicode'))
        self.module.fail_json(**self.result)
    if error != 'EndOfResults':
        self.result['msg'] = "Unexpected search response `%s': %s" % (error, ET.tostring(model_responses, encoding='unicode'))
        self.module.fail_json(**self.result)
    model = model_responses.find('ca:model', self.resp_namespace)
    attrs = model.findall('ca:attribute', self.resp_namespace)
    if not attrs:
        self.result['msg'] = 'No attributes returned.'
        self.module.fail_json(**self.result)
    ret = dict()
    for attr in attrs:
        attr_id = attr.get('id')
        attr_name = self.attr_name(attr_id)
        attr_val = attr.text
        key = attr_name if attr_name in ret_attrs else attr_id
        ret[key] = attr_val
        ret_attrs.remove(key)
    return ret