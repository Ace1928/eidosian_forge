import re
import textwrap
import param
from param.ipython import ParamPager
from param.parameterized import bothmethod
from .util import group_sanitizer, label_sanitizer
@bothmethod
def element_info(cls_or_slf, node, siblings, level, value_dims):
    """
        Return the information summary for an Element. This consists
        of the dotted name followed by an value dimension names.
        """
    info = cls_or_slf.component_type(node)
    if len(node.kdims) >= 1:
        info += cls_or_slf.tab + f'[{','.join((d.name for d in node.kdims))}]'
    if value_dims and len(node.vdims) >= 1:
        info += cls_or_slf.tab + f'({','.join((d.name for d in node.vdims))})'
    return (level, [(level, info)])