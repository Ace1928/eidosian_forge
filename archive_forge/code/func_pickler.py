import os
import tempfile
import pickle
from mpmath import *
def pickler(obj):
    fn = tempfile.mktemp()
    f = open(fn, 'wb')
    pickle.dump(obj, f)
    f.close()
    f = open(fn, 'rb')
    obj2 = pickle.load(f)
    f.close()
    os.remove(fn)
    return obj2