import argparse
from textwrap import indent
import xml.etree.ElementTree as ET
from jeepney.wrappers import Introspectable
from jeepney.io.blocking import open_dbus_connection, Proxy
from jeepney import __version__
from jeepney.wrappers import MessageGenerator, new_method_call
def code_from_xml(xml, path, bus_name, fh):
    if isinstance(fh, (bytes, str)):
        with open(fh, 'w') as f:
            return code_from_xml(xml, path, bus_name, f)
    root = ET.fromstring(xml)
    fh.write(MODULE_TEMPLATE.format(version=__version__, path=path, bus_name=bus_name))
    i = 0
    for interface_node in root.findall('interface'):
        if interface_node.attrib['name'] in IGNORE_INTERFACES:
            continue
        fh.write(Interface(interface_node, path, bus_name).make_code())
        i += 1
    return i