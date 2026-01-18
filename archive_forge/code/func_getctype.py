from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def getctype(var):
    """
    Determines C type
    """
    ctype = 'void'
    if isfunction(var):
        if 'result' in var:
            a = var['result']
        else:
            a = var['name']
        if a in var['vars']:
            return getctype(var['vars'][a])
        else:
            errmess('getctype: function %s has no return value?!\n' % a)
    elif issubroutine(var):
        return ctype
    elif ischaracter_or_characterarray(var):
        return 'character'
    elif isstring_or_stringarray(var):
        return 'string'
    elif 'typespec' in var and var['typespec'].lower() in f2cmap_all:
        typespec = var['typespec'].lower()
        f2cmap = f2cmap_all[typespec]
        ctype = f2cmap['']
        if 'kindselector' in var:
            if '*' in var['kindselector']:
                try:
                    ctype = f2cmap[var['kindselector']['*']]
                except KeyError:
                    errmess('getctype: "%s %s %s" not supported.\n' % (var['typespec'], '*', var['kindselector']['*']))
            elif 'kind' in var['kindselector']:
                if typespec + 'kind' in f2cmap_all:
                    f2cmap = f2cmap_all[typespec + 'kind']
                try:
                    ctype = f2cmap[var['kindselector']['kind']]
                except KeyError:
                    if typespec in f2cmap_all:
                        f2cmap = f2cmap_all[typespec]
                    try:
                        ctype = f2cmap[str(var['kindselector']['kind'])]
                    except KeyError:
                        errmess('getctype: "%s(kind=%s)" is mapped to C "%s" (to override define dict(%s = dict(%s="<C typespec>")) in %s/.f2py_f2cmap file).\n' % (typespec, var['kindselector']['kind'], ctype, typespec, var['kindselector']['kind'], os.getcwd()))
    elif not isexternal(var):
        errmess('getctype: No C-type found in "%s", assuming void.\n' % var)
    return ctype