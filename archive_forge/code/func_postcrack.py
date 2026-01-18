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
def postcrack(block, args=None, tab=''):
    """
    TODO:
          function return values
          determine expression types if in argument list
    """
    global usermodules, onlyfunctions
    if isinstance(block, list):
        gret = []
        uret = []
        for g in block:
            setmesstext(g)
            g = postcrack(g, tab=tab + '\t')
            if 'name' in g and '__user__' in g['name']:
                uret.append(g)
            else:
                gret.append(g)
        return uret + gret
    setmesstext(block)
    if not isinstance(block, dict) and 'block' not in block:
        raise Exception('postcrack: Expected block dictionary instead of ' + str(block))
    if 'name' in block and (not block['name'] == 'unknown_interface'):
        outmess('%sBlock: %s\n' % (tab, block['name']), 0)
    block = analyzeargs(block)
    block = analyzecommon(block)
    block['vars'] = analyzevars(block)
    block['sortvars'] = sortvarnames(block['vars'])
    if 'args' in block and block['args']:
        args = block['args']
    block['body'] = analyzebody(block, args, tab=tab)
    userisdefined = []
    if 'use' in block:
        useblock = block['use']
        for k in list(useblock.keys()):
            if '__user__' in k:
                userisdefined.append(k)
    else:
        useblock = {}
    name = ''
    if 'name' in block:
        name = block['name']
    if 'externals' in block and block['externals']:
        interfaced = []
        if 'interfaced' in block:
            interfaced = block['interfaced']
        mvars = copy.copy(block['vars'])
        if name:
            mname = name + '__user__routines'
        else:
            mname = 'unknown__user__routines'
        if mname in userisdefined:
            i = 1
            while '%s_%i' % (mname, i) in userisdefined:
                i = i + 1
            mname = '%s_%i' % (mname, i)
        interface = {'block': 'interface', 'body': [], 'vars': {}, 'name': name + '_user_interface'}
        for e in block['externals']:
            if e in interfaced:
                edef = []
                j = -1
                for b in block['body']:
                    j = j + 1
                    if b['block'] == 'interface':
                        i = -1
                        for bb in b['body']:
                            i = i + 1
                            if 'name' in bb and bb['name'] == e:
                                edef = copy.copy(bb)
                                del b['body'][i]
                                break
                        if edef:
                            if not b['body']:
                                del block['body'][j]
                            del interfaced[interfaced.index(e)]
                            break
                interface['body'].append(edef)
            elif e in mvars and (not isexternal(mvars[e])):
                interface['vars'][e] = mvars[e]
        if interface['vars'] or interface['body']:
            block['interfaced'] = interfaced
            mblock = {'block': 'python module', 'body': [interface], 'vars': {}, 'name': mname, 'interfaced': block['externals']}
            useblock[mname] = {}
            usermodules.append(mblock)
    if useblock:
        block['use'] = useblock
    return block