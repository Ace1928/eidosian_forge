from _pydev_bundle import pydev_log
import os
from _pydevd_bundle.pydevd_comm import CMD_SIGNATURE_CALL_TRACE, NetCommand
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
def get_type_of_value(value, ignore_module_name=('__main__', '__builtin__', 'builtins'), recursive=False):
    tp = type(value)
    class_name = tp.__name__
    if class_name == 'instance':
        tp = value.__class__
        class_name = tp.__name__
    if hasattr(tp, '__module__') and tp.__module__ and (tp.__module__ not in ignore_module_name):
        class_name = '%s.%s' % (tp.__module__, class_name)
    if class_name == 'list':
        class_name = 'List'
        if len(value) > 0 and recursive:
            class_name += '[%s]' % get_type_of_value(value[0], recursive=recursive)
        return class_name
    if class_name == 'dict':
        class_name = 'Dict'
        if len(value) > 0 and recursive:
            for k, v in value.items():
                class_name += '[%s, %s]' % (get_type_of_value(k, recursive=recursive), get_type_of_value(v, recursive=recursive))
                break
        return class_name
    if class_name == 'tuple':
        class_name = 'Tuple'
        if len(value) > 0 and recursive:
            class_name += '['
            class_name += ', '.join((get_type_of_value(v, recursive=recursive) for v in value))
            class_name += ']'
    return class_name