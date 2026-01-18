import re
import textwrap
import param
from param.ipython import ParamPager
from param.parameterized import bothmethod
from .util import group_sanitizer, label_sanitizer
@bothmethod
def ndmapping_info(cls_or_slf, node, siblings, level, value_dims):
    key_dim_info = f'[{','.join((d.name for d in node.kdims))}]'
    first_line = cls_or_slf.component_type(node) + cls_or_slf.tab + key_dim_info
    lines = [(level, first_line)]
    opts = cls_or_slf.option_info(node)
    if cls_or_slf.show_options and opts and opts.kwargs:
        lines += [(level, l) for l in cls_or_slf.format_options(opts)]
    if len(node.data) == 0:
        return (level, lines)
    last = list(node.data.values())[-1]
    if last is not None and last._deep_indexable and (not hasattr(last, 'children')):
        level, additional_lines = cls_or_slf.ndmapping_info(last, [], level, value_dims)
    else:
        additional_lines = cls_or_slf.recurse(last, level=level, value_dims=value_dims)
    lines += cls_or_slf.shift(additional_lines, 1)
    return (level, lines)