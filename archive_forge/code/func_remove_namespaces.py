from lxml import etree
from .default import DefaultDeviceHandler
from ncclient.operations.third_party.alu.rpc import GetConfiguration, LoadConfiguration, ShowCLI
from ncclient.xml_ import BASE_NS_1_0
def remove_namespaces(xml):
    for elem in xml.getiterator():
        if elem.tag is etree.Comment:
            continue
        i = elem.tag.find('}')
        if i > 0:
            elem.tag = elem.tag[i + 1:]
    etree.cleanup_namespaces(xml)
    return xml