from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def repr_regex(regex):
    all_flags = ('A', 'I', 'DEBUG', 'L', 'M', 'S', 'X')
    flags = ' | '.join((f're.{f}' for f in all_flags if regex.flags & getattr(re, f)))
    flags = ', ' + flags if flags else ''
    return 're.compile({!r}{})'.format(regex.pattern, flags)