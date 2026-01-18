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
def handleRequest(self):
    """Handle a single request from the remote process. 
        Blocks until a request is available."""
    result = None
    while True:
        try:
            cmd, reqId, nByteMsgs, optStr = self.conn.recv()
            break
        except EOFError:
            self.debugMsg('  handleRequest: got EOFError from recv; raise ClosedError.')
            raise ClosedError()
        except IOError as err:
            if err.errno == 4:
                self.debugMsg('  handleRequest: got IOError 4 from recv; try again.')
                continue
            else:
                self.debugMsg('  handleRequest: got IOError %d from recv (%s); raise ClosedError.', err.errno, err.strerror)
                raise ClosedError()
    self.debugMsg('  handleRequest: received %s %s', cmd, reqId)
    byteData = []
    if nByteMsgs > 0:
        self.debugMsg('    handleRequest: reading %d byte messages', nByteMsgs)
    for i in range(nByteMsgs):
        while True:
            try:
                byteData.append(self.conn.recv_bytes())
                break
            except EOFError:
                self.debugMsg('    handleRequest: got EOF while reading byte messages; raise ClosedError.')
                raise ClosedError()
            except IOError as err:
                if err.errno == 4:
                    self.debugMsg('    handleRequest: got IOError 4 while reading byte messages; try again.')
                    continue
                else:
                    self.debugMsg('    handleRequest: got IOError while reading byte messages; raise ClosedError.')
                    raise ClosedError()
    try:
        if cmd == 'result' or cmd == 'error':
            resultId = reqId
            reqId = None
        opts = pickle.loads(optStr)
        self.debugMsg('    handleRequest: id=%s opts=%s', reqId, opts)
        returnType = opts.get('returnType', 'auto')
        if cmd == 'result':
            with self.resultLock:
                self.results[resultId] = ('result', opts['result'])
        elif cmd == 'error':
            with self.resultLock:
                self.results[resultId] = ('error', (opts['exception'], opts['excString']))
        elif cmd == 'getObjAttr':
            result = getattr(opts['obj'], opts['attr'])
        elif cmd == 'callObj':
            obj = opts['obj']
            fnargs = opts['args']
            fnkwds = opts['kwds']
            if len(byteData) > 0:
                for i, arg in enumerate(fnargs):
                    if isinstance(arg, tuple) and len(arg) > 0 and (arg[0] == '__byte_message__'):
                        ind = arg[1]
                        dtype, shape = arg[2]
                        fnargs[i] = np.frombuffer(byteData[ind], dtype=dtype).reshape(shape)
                for k, arg in fnkwds.items():
                    if isinstance(arg, tuple) and len(arg) > 0 and (arg[0] == '__byte_message__'):
                        ind = arg[1]
                        dtype, shape = arg[2]
                        fnkwds[k] = np.frombuffer(byteData[ind], dtype=dtype).reshape(shape)
            if len(fnkwds) == 0:
                try:
                    result = obj(*fnargs)
                except:
                    print('Failed to call object %s: %d, %s' % (obj, len(fnargs), fnargs[1:]))
                    raise
            else:
                result = obj(*fnargs, **fnkwds)
        elif cmd == 'getObjValue':
            result = opts['obj']
            returnType = 'value'
        elif cmd == 'transfer':
            result = opts['obj']
            returnType = 'proxy'
        elif cmd == 'transferArray':
            result = np.frombuffer(byteData[0], dtype=opts['dtype']).reshape(opts['shape'])
            returnType = 'proxy'
        elif cmd == 'import':
            name = opts['module']
            fromlist = opts.get('fromlist', [])
            mod = builtins.__import__(name, fromlist=fromlist)
            if len(fromlist) == 0:
                parts = name.lstrip('.').split('.')
                result = mod
                for part in parts[1:]:
                    result = getattr(result, part)
            else:
                result = map(mod.__getattr__, fromlist)
        elif cmd == 'del':
            LocalObjectProxy.releaseProxyId(opts['proxyId'])
        elif cmd == 'close':
            if reqId is not None:
                result = True
                returnType = 'value'
        exc = None
    except:
        exc = sys.exc_info()
    if reqId is not None:
        if exc is None:
            self.debugMsg('    handleRequest: sending return value for %d: %s', reqId, result)
            if returnType == 'auto':
                with self.optsLock:
                    noProxyTypes = self.proxyOptions['noProxyTypes']
                result = self.autoProxy(result, noProxyTypes)
            elif returnType == 'proxy':
                result = LocalObjectProxy(result)
            try:
                self.replyResult(reqId, result)
            except:
                sys.excepthook(*sys.exc_info())
                self.replyError(reqId, *sys.exc_info())
        else:
            self.debugMsg('    handleRequest: returning exception for %d', reqId)
            self.replyError(reqId, *exc)
    elif exc is not None:
        sys.excepthook(*exc)
    if cmd == 'close':
        if opts.get('noCleanup', False) is True:
            os._exit(0)
        else:
            raise ClosedError()