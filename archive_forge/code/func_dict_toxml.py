import datetime
import xml.etree.ElementTree as et
from simplegeneric import generic
import wsme.types
from wsme.exc import UnknownArgument, InvalidInput
import re
@toxml.when_type(wsme.types.DictType)
def dict_toxml(datatype, key, value):
    el = et.Element(key)
    if value is None:
        el.set('nil', 'true')
    else:
        for item in value.items():
            key = toxml(datatype.key_type, 'key', item[0])
            value = toxml(datatype.value_type, 'value', item[1])
            node = et.Element('item')
            node.append(key)
            node.append(value)
            el.append(node)
    return el