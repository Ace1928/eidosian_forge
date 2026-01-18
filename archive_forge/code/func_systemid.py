from __future__ import absolute_import, division, print_function
import os
import sys
from ansible_collections.community.general.plugins.module_utils import redhat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import urllib, xmlrpc_client
@property
def systemid(self):
    systemid = None
    xpath_str = "//member[name='system_id']/value/string"
    if os.path.isfile(self.config['systemIdPath']):
        fd = open(self.config['systemIdPath'], 'r')
        xml_data = fd.read()
        fd.close()
        if systemid is None:
            try:
                import libxml2
                doc = libxml2.parseDoc(xml_data)
                ctxt = doc.xpathNewContext()
                systemid = ctxt.xpathEval(xpath_str)[0].content
                doc.freeDoc()
                ctxt.xpathFreeContext()
            except ImportError:
                pass
        if systemid is None:
            try:
                from lxml import etree
                root = etree.fromstring(xml_data)
                systemid = root.xpath(xpath_str)[0].text
            except ImportError:
                raise Exception('"libxml2" or "lxml" is required for this module.')
        if systemid is not None and systemid.startswith('ID-'):
            systemid = systemid[3:]
    return int(systemid)