import os
import sys
import re
def parse_values(astr):
    astr = parenrep.sub(paren_repl, astr)
    astr = ','.join([plainrep.sub(paren_repl, x.strip()) for x in astr.split(',')])
    return astr.split(',')