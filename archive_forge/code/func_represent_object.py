from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_object(self, data):
    cls = type(data)
    if cls in copyreg.dispatch_table:
        reduce = copyreg.dispatch_table[cls](data)
    elif hasattr(data, '__reduce_ex__'):
        reduce = data.__reduce_ex__(2)
    elif hasattr(data, '__reduce__'):
        reduce = data.__reduce__()
    else:
        raise RepresenterError('cannot represent an object', data)
    reduce = (list(reduce) + [None] * 5)[:5]
    function, args, state, listitems, dictitems = reduce
    args = list(args)
    if state is None:
        state = {}
    if listitems is not None:
        listitems = list(listitems)
    if dictitems is not None:
        dictitems = dict(dictitems)
    if function.__name__ == '__newobj__':
        function = args[0]
        args = args[1:]
        tag = 'tag:yaml.org,2002:python/object/new:'
        newobj = True
    else:
        tag = 'tag:yaml.org,2002:python/object/apply:'
        newobj = False
    function_name = '%s.%s' % (function.__module__, function.__name__)
    if not args and (not listitems) and (not dictitems) and isinstance(state, dict) and newobj:
        return self.represent_mapping('tag:yaml.org,2002:python/object:' + function_name, state)
    if not listitems and (not dictitems) and isinstance(state, dict) and (not state):
        return self.represent_sequence(tag + function_name, args)
    value = {}
    if args:
        value['args'] = args
    if state or not isinstance(state, dict):
        value['state'] = state
    if listitems:
        value['listitems'] = listitems
    if dictitems:
        value['dictitems'] = dictitems
    return self.represent_mapping(tag + function_name, value)