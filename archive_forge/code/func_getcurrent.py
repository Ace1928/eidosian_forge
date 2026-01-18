import sys
import types
import stackless
def getcurrent():
    return tasklet_to_greenlet[stackless.getcurrent()]