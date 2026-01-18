import datetime
import xml.etree.ElementTree as et
from simplegeneric import generic
import wsme.types
from wsme.exc import UnknownArgument, InvalidInput
import re
@fromxml.when_type(wsme.types.DictType)
def dict_fromxml(datatype, element):
    if element.get('nil') == 'true':
        return None
    return dict(((fromxml(datatype.key_type, item.find('key')), fromxml(datatype.value_type, item.find('value'))) for item in element.findall('item')))