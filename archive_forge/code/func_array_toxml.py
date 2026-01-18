import datetime
import xml.etree.ElementTree as et
from simplegeneric import generic
import wsme.types
from wsme.exc import UnknownArgument, InvalidInput
import re
@toxml.when_type(wsme.types.ArrayType)
def array_toxml(datatype, key, value):
    el = et.Element(key)
    if value is None:
        el.set('nil', 'true')
    else:
        for item in value:
            el.append(toxml(datatype.item_type, 'item', item))
    return el