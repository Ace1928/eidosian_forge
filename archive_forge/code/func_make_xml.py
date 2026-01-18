import doctest
import xml.etree.ElementTree as ET
def make_xml(s):
    return ET.XML('<xml>%s</xml>' % s)