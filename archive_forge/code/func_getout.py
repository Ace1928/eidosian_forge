import os
from os.path import sep
def getout(*args):
    try:
        return Popen(args, stdout=PIPE).communicate()[0]
    except OSError:
        return ''