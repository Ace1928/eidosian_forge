import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def onResult(success, result):
    ctx = context.theContextTracker.currentContext().contexts[-1]
    contexts.append(ctx)
    event.set()