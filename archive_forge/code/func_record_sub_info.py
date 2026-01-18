from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def record_sub_info(match_object, sub_info=sub_info):
    sub_info.append([match_object.group(1)[0], match_object.span()])
    return match_object.group(1)