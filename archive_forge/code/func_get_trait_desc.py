import os
from contextlib import AbstractContextManager
from copy import deepcopy
from textwrap import wrap
import re
from datetime import datetime as dt
from dateutil.parser import parse as parseutc
import platform
from ... import logging, config
from ...utils.misc import is_container, rgetcwd
from ...utils.filemanip import md5, hash_infile
def get_trait_desc(inputs, name, spec):
    """Parses a HasTraits object into a nipype documentation string"""
    desc = spec.desc
    xor = spec.xor
    requires = spec.requires
    argstr = spec.argstr
    manhelpstr = ['\t%s' % name]
    type_info = spec.full_info(inputs, name, None)
    default = ''
    if spec.usedefault:
        default = ', nipype default value: %s' % str(spec.default_value()[1])
    line = '(%s%s)' % (type_info, default)
    manhelpstr = wrap(line, HELP_LINEWIDTH, initial_indent=manhelpstr[0] + ': ', subsequent_indent='\t\t  ')
    if desc:
        for line in desc.split('\n'):
            line = re.sub('\\s+', ' ', line)
            manhelpstr += wrap(line, HELP_LINEWIDTH, initial_indent='\t\t', subsequent_indent='\t\t')
    if argstr:
        pos = spec.position
        if pos is not None:
            manhelpstr += wrap('argument: ``%s``, position: %s' % (argstr, pos), HELP_LINEWIDTH, initial_indent='\t\t', subsequent_indent='\t\t')
        else:
            manhelpstr += wrap('argument: ``%s``' % argstr, HELP_LINEWIDTH, initial_indent='\t\t', subsequent_indent='\t\t')
    if xor:
        line = '%s' % ', '.join(xor)
        manhelpstr += wrap(line, HELP_LINEWIDTH, initial_indent='\t\tmutually_exclusive: ', subsequent_indent='\t\t  ')
    if requires:
        others = [field for field in requires if field != name]
        line = '%s' % ', '.join(others)
        manhelpstr += wrap(line, HELP_LINEWIDTH, initial_indent='\t\trequires: ', subsequent_indent='\t\t  ')
    return manhelpstr