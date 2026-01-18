from __future__ import absolute_import, division, print_function
import ssl
import atexit
import base64
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils._text import to_text, to_native
def parse_vim_property(vim_prop):
    """
    Helper method to parse VIM properties of virtual machine
    """
    prop_type = type(vim_prop).__name__
    if prop_type.startswith(('vim', 'vmodl', 'Link')):
        if isinstance(vim_prop, DataObject):
            r = {}
            for prop in vim_prop._GetPropertyList():
                if prop.name not in ['dynamicProperty', 'dynamicType', 'managedObjectType']:
                    sub_prop = getattr(vim_prop, prop.name)
                    r[prop.name] = parse_vim_property(sub_prop)
            return r
        if isinstance(vim_prop, list):
            r = []
            for prop in vim_prop:
                r.append(parse_vim_property(prop))
            return r
        return vim_prop.__str__()
    elif prop_type == 'datetime':
        return Iso8601.ISO8601Format(vim_prop)
    elif prop_type == 'long':
        return int(vim_prop)
    elif prop_type == 'long[]':
        return [int(x) for x in vim_prop]
    elif isinstance(vim_prop, list):
        return [parse_vim_property(x) for x in vim_prop]
    elif prop_type in ['bool', 'int', 'NoneType', 'dict']:
        return vim_prop
    elif prop_type in ['binary']:
        return to_text(base64.b64encode(vim_prop))
    return to_text(vim_prop)