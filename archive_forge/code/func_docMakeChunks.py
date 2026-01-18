from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def docMakeChunks(optList, width=80):
    """
    Makes doc chunks for option declarations.

    Takes a list of dictionaries, each of which may have one or more
    of the keys 'long', 'short', 'doc', 'default', 'optType'.

    Returns a list of strings.
    The strings may be multiple lines,
    all of them end with a newline.
    """
    optList.sort(key=lambda o: o.get('short', None) or o.get('long', None))
    maxOptLen = 0
    for opt in optList:
        optLen = len(opt.get('long', ''))
        if optLen:
            if opt.get('optType', None) == 'parameter':
                optLen = optLen + 1
            maxOptLen = max(optLen, maxOptLen)
    colWidth1 = maxOptLen + len('  -s, --  ')
    colWidth2 = width - colWidth1
    colFiller1 = ' ' * colWidth1
    optChunks = []
    seen = {}
    for opt in optList:
        if opt.get('short', None) in seen or opt.get('long', None) in seen:
            continue
        for x in (opt.get('short', None), opt.get('long', None)):
            if x is not None:
                seen[x] = 1
        optLines = []
        comma = ' '
        if opt.get('short', None):
            short = '-%c' % (opt['short'],)
        else:
            short = ''
        if opt.get('long', None):
            long = opt['long']
            if opt.get('optType', None) == 'parameter':
                long = long + '='
            long = '%-*s' % (maxOptLen, long)
            if short:
                comma = ','
        else:
            long = ' ' * (maxOptLen + len('--'))
        if opt.get('optType', None) == 'command':
            column1 = '    %s      ' % long
        else:
            column1 = '  %2s%c --%s  ' % (short, comma, long)
        if opt.get('doc', ''):
            doc = opt['doc'].strip()
        else:
            doc = ''
        if opt.get('optType', None) == 'parameter' and (not opt.get('default', None) is None):
            doc = '{} [default: {}]'.format(doc, opt['default'])
        if opt.get('optType', None) == 'parameter' and opt.get('dispatch', None) is not None:
            d = opt['dispatch']
            if isinstance(d, CoerceParameter) and d.doc:
                doc = f'{doc}. {d.doc}'
        if doc:
            column2_l = textwrap.wrap(doc, colWidth2)
        else:
            column2_l = ['']
        optLines.append(f'{column1}{column2_l.pop(0)}\n')
        for line in column2_l:
            optLines.append(f'{colFiller1}{line}\n')
        optChunks.append(''.join(optLines))
    return optChunks