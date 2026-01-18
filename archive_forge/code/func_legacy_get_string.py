from typing import Optional
import warnings
def legacy_get_string(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool=True) -> str:
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret