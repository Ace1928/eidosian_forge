import logging
from oslo_utils import timeutils
from suds import sudsobject
def get_moref(value, type_):
    """Get managed object reference.

    :param value: value of the managed object
    :param type_: type of the managed object
    :returns: managed object reference with given value and type
    """
    moref = sudsobject.Property(value)
    moref._type = type_
    return moref