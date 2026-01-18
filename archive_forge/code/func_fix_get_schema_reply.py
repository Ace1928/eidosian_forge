import logging
import re
from lxml import etree
from lxml.etree import QName
from ncclient.operations.retrieve import GetSchemaReply
from .default import DefaultDeviceHandler
from ncclient.operations.third_party.juniper.rpc import GetConfiguration, LoadConfiguration, CompareConfiguration
from ncclient.operations.third_party.juniper.rpc import ExecuteRpc, Command, Reboot, Halt, Commit, Rollback
from ncclient.operations.rpc import RPCError
from ncclient.xml_ import to_ele, replace_namespace, BASE_NS_1_0, NETCONF_MONITORING_NS
from ncclient.transport.third_party.junos.parser import JunosXMLParser
from ncclient.transport.parser import DefaultXMLParser
from ncclient.transport.parser import SAXParserHandler
def fix_get_schema_reply(root):
    data_elems = root.xpath('/nc:rpc-reply/*[local-name()="data"]', namespaces={'nc': BASE_NS_1_0})
    if len(data_elems) != 1:
        return
    data_el = data_elems[0]
    namespace = QName(data_el).namespace
    if namespace == BASE_NS_1_0:
        logger.warning("The device seems to run non-rfc compliant netconf. You may want to configure: 'set system services netconf rfc-compliant'")
        replace_namespace(data_el, old_ns=BASE_NS_1_0, new_ns=NETCONF_MONITORING_NS)
    elif namespace is None:
        replace_namespace(data_el, old_ns=None, new_ns=NETCONF_MONITORING_NS)