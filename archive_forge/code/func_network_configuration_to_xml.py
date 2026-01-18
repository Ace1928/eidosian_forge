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
def network_configuration_to_xml(configuration, xml):
    AzureXmlSerializer.data_to_xml([('ConfigurationSetType', configuration.configuration_set_type)], xml)
    input_endpoints = ET.Element('InputEndpoints')
    xml.append(input_endpoints)
    for endpoint in configuration.input_endpoints:
        input_endpoint = ET.Element('InputEndpoint')
        input_endpoints.append(input_endpoint)
        AzureXmlSerializer.data_to_xml([('LoadBalancedEndpointSetName', endpoint.load_balanced_endpoint_set_name)], input_endpoint)
        AzureXmlSerializer.data_to_xml([('LocalPort', endpoint.local_port)], input_endpoint)
        AzureXmlSerializer.data_to_xml([('Name', endpoint.name)], input_endpoint)
        AzureXmlSerializer.data_to_xml([('Port', endpoint.port)], input_endpoint)
        if endpoint.load_balancer_probe.path or endpoint.load_balancer_probe.port or endpoint.load_balancer_probe.protocol:
            load_balancer_probe = ET.Element('LoadBalancerProbe')
            input_endpoint.append(load_balancer_probe)
            AzureXmlSerializer.data_to_xml([('Path', endpoint.load_balancer_probe.path)], load_balancer_probe)
            AzureXmlSerializer.data_to_xml([('Port', endpoint.load_balancer_probe.port)], load_balancer_probe)
            AzureXmlSerializer.data_to_xml([('Protocol', endpoint.load_balancer_probe.protocol)], load_balancer_probe)
        AzureXmlSerializer.data_to_xml([('Protocol', endpoint.protocol)], input_endpoint)
        AzureXmlSerializer.data_to_xml([('EnableDirectServerReturn', endpoint.enable_direct_server_return, _lower)], input_endpoint)
    subnet_names = ET.Element('SubnetNames')
    xml.append(subnet_names)
    for name in configuration.subnet_names:
        AzureXmlSerializer.data_to_xml([('SubnetName', name)], subnet_names)
    return xml