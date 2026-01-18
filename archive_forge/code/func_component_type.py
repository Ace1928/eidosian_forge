import re
import textwrap
import param
from param.ipython import ParamPager
from param.parameterized import bothmethod
from .util import group_sanitizer, label_sanitizer
@bothmethod
def component_type(cls_or_slf, node):
    """Return the type.group.label dotted information"""
    if node is None:
        return ''
    return cls_or_slf.type_formatter.format(type=str(type(node).__name__))