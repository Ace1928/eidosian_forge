import cgi
import datetime
import re
from simplegeneric import generic
from wsme.exc import ClientSideError, UnknownArgument, InvalidInput
from wsme.types import iscomplex, list_attributes, Unset
from wsme.types import UserType, ArrayType, DictType, File
from wsme.utils import parse_isodate, parse_isotime, parse_isodatetime
import wsme.runtime
@from_param.when_object(File)
def filetype_from_param(datatype, value):
    if isinstance(value, cgi.FieldStorage):
        return File(fieldstorage=value)
    return File(content=value)