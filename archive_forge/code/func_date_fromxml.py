import datetime
import xml.etree.ElementTree as et
from simplegeneric import generic
import wsme.types
from wsme.exc import UnknownArgument, InvalidInput
import re
@fromxml.when_object(datetime.date)
def date_fromxml(datatype, element):
    if element.get('nil') == 'true':
        return None
    return wsme.utils.parse_isodate(element.text)