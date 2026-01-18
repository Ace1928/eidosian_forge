import sys
from pyasn1 import error
from pyasn1.compat import calling
from pyasn1.type import constraint
from pyasn1.type import tag
from pyasn1.type import tagmap
@classmethod
def getTypeId(cls, increment=1):
    try:
        Asn1Item._typeCounter += increment
    except AttributeError:
        Asn1Item._typeCounter = increment
    return Asn1Item._typeCounter