import datetime
import xml.etree.ElementTree as et
from simplegeneric import generic
import wsme.types
from wsme.exc import UnknownArgument, InvalidInput
import re
@fromxml.when_object(datetime.datetime)
def datetime_fromxml(datatype, element):
    if element.get('nil') == 'true':
        return None
    return wsme.utils.parse_isodatetime(element.text)