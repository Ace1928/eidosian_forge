import codecs
import hashlib
import io
import json
import os
import sys
import atexit
import shutil
import tempfile
def elt(eltType, numargs):

    def fun(*args):
        lenargs = len(args)
        if lenargs != numargs:
            raise ValueError(eltType + ' expects ' + str(numargs) + ' arguments, but given ' + str(lenargs))
        if numargs == 0:
            xs = []
        elif len(args) == 1:
            xs = args[0]
        else:
            xs = list(args)
        return {'t': eltType, 'c': xs}
    return fun