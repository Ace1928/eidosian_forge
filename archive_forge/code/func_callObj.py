import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint
def callObj(self, obj, args, kwds, **opts):
    opts = opts.copy()
    args = list(args)
    with self.optsLock:
        noProxyTypes = opts.pop('noProxyTypes', None)
        if noProxyTypes is None:
            noProxyTypes = self.proxyOptions['noProxyTypes']
        autoProxy = opts.pop('autoProxy', self.proxyOptions['autoProxy'])
    if autoProxy is True:
        args = [self.autoProxy(v, noProxyTypes) for v in args]
        for k, v in kwds.items():
            opts[k] = self.autoProxy(v, noProxyTypes)
    byteMsgs = []
    for i, arg in enumerate(args):
        if arg.__class__ == np.ndarray:
            args[i] = ('__byte_message__', len(byteMsgs), (arg.dtype, arg.shape))
            byteMsgs.append(arg)
    for k, v in kwds.items():
        if v.__class__ == np.ndarray:
            kwds[k] = ('__byte_message__', len(byteMsgs), (v.dtype, v.shape))
            byteMsgs.append(v)
    return self.send(request='callObj', opts=dict(obj=obj, args=args, kwds=kwds), byteData=byteMsgs, **opts)