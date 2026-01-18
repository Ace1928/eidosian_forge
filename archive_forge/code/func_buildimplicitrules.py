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
def buildimplicitrules(block):
    setmesstext(block)
    implicitrules = defaultimplicitrules
    attrrules = {}
    if 'implicit' in block:
        if block['implicit'] is None:
            implicitrules = None
            if verbose > 1:
                outmess('buildimplicitrules: no implicit rules for routine %s.\n' % repr(block['name']))
        else:
            for k in list(block['implicit'].keys()):
                if block['implicit'][k].get('typespec') not in ['static', 'automatic']:
                    implicitrules[k] = block['implicit'][k]
                else:
                    attrrules[k] = block['implicit'][k]['typespec']
    return (implicitrules, attrrules)