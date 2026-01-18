from lxml import etree
from .default import DefaultDeviceHandler
from ncclient.operations.third_party.sros.rpc import MdCliRawCommand, Commit
from ncclient.xml_ import BASE_NS_1_0
def get_xml_base_namespace_dict(self):
    return {None: BASE_NS_1_0}