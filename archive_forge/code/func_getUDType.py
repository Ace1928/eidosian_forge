from pyparsing import *
def getUDType(typestr):
    key = typestr.rstrip(' *')
    if key not in typemap:
        user_defined_types.add(key)
        typemap[key] = '{0}_{1}'.format(module, key)