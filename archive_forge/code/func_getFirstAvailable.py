from subprocess import Popen, PIPE
from distutils import spawn
import os
import math
import random
import time
import sys
import platform
def getFirstAvailable(order='first', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False, includeNan=False, excludeID=[], excludeUUID=[]):
    for i in range(attempts):
        if verbose:
            print('Attempting (' + str(i + 1) + '/' + str(attempts) + ') to locate available GPU.')
        available = getAvailable(order=order, limit=1, maxLoad=maxLoad, maxMemory=maxMemory, includeNan=includeNan, excludeID=excludeID, excludeUUID=excludeUUID)
        if available:
            if verbose:
                print('GPU ' + str(available) + ' located!')
            break
        if i != attempts - 1:
            time.sleep(interval)
    if not available:
        raise RuntimeError('Could not find an available GPU after ' + str(attempts) + ' attempts with ' + str(interval) + ' seconds interval.')
    return available