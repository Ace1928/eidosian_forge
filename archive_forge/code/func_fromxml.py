import datetime
import xml.etree.ElementTree as et
from simplegeneric import generic
import wsme.types
from wsme.exc import UnknownArgument, InvalidInput
import re
@generic
def fromxml(datatype, element):
    """
    A generic converter from xml elements to python datatype.

    If a non-complex user specific type is to be used in the api,
    a specific fromxml should be added::

        from wsme.protocol.restxml import fromxml

        class MySpecialType(object):
            pass

        @fromxml.when_object(MySpecialType)
        def myspecialtype_fromxml(datatype, element):
            if element.get('nil', False):
                return None
            return MySpecialType(element.text)
    """
    if element.get('nil', False):
        return None
    if wsme.types.isusertype(datatype):
        return datatype.frombasetype(fromxml(datatype.basetype, element))
    if wsme.types.iscomplex(datatype):
        obj = datatype()
        for attrdef in wsme.types.list_attributes(datatype):
            sub = element.find(attrdef.name)
            if sub is not None:
                val_fromxml = fromxml(attrdef.datatype, sub)
                if getattr(attrdef, 'readonly', False):
                    raise InvalidInput(attrdef.name, val_fromxml, 'Cannot set read only field.')
                setattr(obj, attrdef.key, val_fromxml)
            elif attrdef.mandatory:
                raise InvalidInput(attrdef.name, None, 'Mandatory field missing.')
        return wsme.types.validate_value(datatype, obj)
    if datatype is wsme.types.bytes:
        return element.text.encode('ascii')
    return datatype(element.text)