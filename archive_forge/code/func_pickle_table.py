import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def pickle_table(self, filename, signature=''):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(filename, 'wb') as outf:
        pickle.dump(__tabversion__, outf, pickle_protocol)
        pickle.dump(self.lr_method, outf, pickle_protocol)
        pickle.dump(signature, outf, pickle_protocol)
        pickle.dump(self.lr_action, outf, pickle_protocol)
        pickle.dump(self.lr_goto, outf, pickle_protocol)
        outp = []
        for p in self.lr_productions:
            if p.func:
                outp.append((p.str, p.name, p.len, p.func, os.path.basename(p.file), p.line))
            else:
                outp.append((str(p), p.name, p.len, None, None, None))
        pickle.dump(outp, outf, pickle_protocol)