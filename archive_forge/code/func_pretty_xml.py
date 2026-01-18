from lxml import etree
from testtools import matchers
def pretty_xml(xml):
    parser = etree.XMLParser(remove_blank_text=True)
    doc = etree.fromstring(xml.strip(), parser)
    return etree.tostring(doc, encoding='utf-8', pretty_print=True).decode('utf-8')