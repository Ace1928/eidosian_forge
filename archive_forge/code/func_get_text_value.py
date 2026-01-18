import boto
from boto.utils import find_class, Password
from boto.sdb.db.key import Key
from boto.sdb.db.model import Model
from boto.compat import six, encodebytes
from datetime import datetime
from xml.dom.minidom import getDOMImplementation, parse, parseString, Node
def get_text_value(self, parent_node):
    value = ''
    for node in parent_node.childNodes:
        if node.nodeType == node.TEXT_NODE:
            value += node.data
    return value