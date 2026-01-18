import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
@staticmethod
def capture_role_to_xml(post_capture_action, target_image_name, target_image_label, provisioning_configuration):
    xml = AzureXmlSerializer.data_to_xml([('OperationType', 'CaptureRoleOperation')])
    AzureXmlSerializer.data_to_xml([('PostCaptureAction', post_capture_action)], xml)
    if provisioning_configuration is not None:
        provisioning_config = ET.Element('ProvisioningConfiguration')
        xml.append(provisioning_config)
        if isinstance(provisioning_configuration, WindowsConfigurationSet):
            AzureXmlSerializer.windows_configuration_to_xml(provisioning_configuration, provisioning_config)
        elif isinstance(provisioning_configuration, LinuxConfigurationSet):
            AzureXmlSerializer.linux_configuration_to_xml(provisioning_configuration, provisioning_config)
    AzureXmlSerializer.data_to_xml([('TargetImageLabel', target_image_label)], xml)
    AzureXmlSerializer.data_to_xml([('TargetImageName', target_image_name)], xml)
    doc = AzureXmlSerializer.doc_from_xml('CaptureRoleOperation', xml)
    result = ensure_string(ET.tostring(doc, encoding='utf-8'))
    return result