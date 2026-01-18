import argparse
from textwrap import indent
import xml.etree.ElementTree as ET
from jeepney.wrappers import Introspectable
from jeepney.io.blocking import open_dbus_connection, Proxy
from jeepney import __version__
from jeepney.wrappers import MessageGenerator, new_method_call
def make_code(self):
    cls_name = self.name.split('.')[-1]
    chunks = [INTERFACE_CLASS_TEMPLATE.format(cls_name=cls_name, interface=self.name, path=self.path, bus_name=self.bus_name)]
    for method in self.methods:
        chunks.append(indent(method.make_code(), ' ' * 4))
    return '\n'.join(chunks)