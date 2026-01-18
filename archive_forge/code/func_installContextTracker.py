from threading import local
from typing import Dict, Type
def installContextTracker(ctr):
    global theContextTracker
    global call
    global get
    theContextTracker = ctr
    call = theContextTracker.callWithContext
    get = theContextTracker.getContext