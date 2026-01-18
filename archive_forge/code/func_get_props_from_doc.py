import boto
from boto.utils import find_class, Password
from boto.sdb.db.key import Key
from boto.sdb.db.model import Model
from boto.compat import six, encodebytes
from datetime import datetime
from xml.dom.minidom import getDOMImplementation, parse, parseString, Node
def get_props_from_doc(self, cls, id, doc):
    """
        Pull out the properties from this document
        Returns the class, the properties in a hash, and the id if provided as a tuple
        :return: (cls, props, id)
        """
    obj_node = doc.getElementsByTagName('object')[0]
    if not cls:
        class_name = obj_node.getAttribute('class')
        cls = find_class(class_name)
    if not id:
        id = obj_node.getAttribute('id')
    props = {}
    for prop_node in obj_node.getElementsByTagName('property'):
        prop_name = prop_node.getAttribute('name')
        prop = cls.find_property(prop_name)
        value = self.decode_value(prop, prop_node)
        value = prop.make_value_from_datastore(value)
        if value is not None:
            props[prop.name] = value
    return (cls, props, id)