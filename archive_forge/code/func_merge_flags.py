import sys, os, subprocess
from .error import PkgConfigError
def merge_flags(cfg1, cfg2):
    """Merge values from cffi config flags cfg2 to cf1

    Example:
        merge_flags({"libraries": ["one"]}, {"libraries": ["two"]})
        {"libraries": ["one", "two"]}
    """
    for key, value in cfg2.items():
        if key not in cfg1:
            cfg1[key] = value
        else:
            if not isinstance(cfg1[key], list):
                raise TypeError('cfg1[%r] should be a list of strings' % (key,))
            if not isinstance(value, list):
                raise TypeError('cfg2[%r] should be a list of strings' % (key,))
            cfg1[key].extend(value)
    return cfg1