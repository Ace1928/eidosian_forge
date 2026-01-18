import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def analyzebody(block, args, tab=''):
    global usermodules, skipfuncs, onlyfuncs, f90modulevars
    setmesstext(block)
    maybe_private = {key: value for key, value in block['vars'].items() if 'attrspec' not in value or 'public' not in value['attrspec']}
    body = []
    for b in block['body']:
        b['parent_block'] = block
        if b['block'] in ['function', 'subroutine']:
            if args is not None and b['name'] not in args:
                continue
            else:
                as_ = b['args']
            if b['name'] in maybe_private.keys():
                skipfuncs.append(b['name'])
            if b['name'] in skipfuncs:
                continue
            if onlyfuncs and b['name'] not in onlyfuncs:
                continue
            b['saved_interface'] = crack2fortrangen(b, '\n' + ' ' * 6, as_interface=True)
        else:
            as_ = args
        b = postcrack(b, as_, tab=tab + '\t')
        if b['block'] in ['interface', 'abstract interface'] and (not b['body']) and (not b.get('implementedby')):
            if 'f2pyenhancements' not in b:
                continue
        if b['block'].replace(' ', '') == 'pythonmodule':
            usermodules.append(b)
        else:
            if b['block'] == 'module':
                f90modulevars[b['name']] = b['vars']
            body.append(b)
    return body