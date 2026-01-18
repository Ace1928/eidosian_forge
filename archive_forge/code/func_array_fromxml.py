import datetime
import xml.etree.ElementTree as et
from simplegeneric import generic
import wsme.types
from wsme.exc import UnknownArgument, InvalidInput
import re
@fromxml.when_type(wsme.types.ArrayType)
def array_fromxml(datatype, element):
    if element.get('nil') == 'true':
        return None
    return [fromxml(datatype.item_type, item) for item in element.findall('item')]