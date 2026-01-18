import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
def register_parallel_cleanup_function():
    """Call MPI_Abort if python crashes.

    This will terminate the processes on the other nodes."""
    if world.size == 1:
        return

    def cleanup(sys=sys, time=time, world=world):
        error = getattr(sys, 'last_type', None)
        if error:
            sys.stdout.flush()
            sys.stderr.write(('ASE CLEANUP (node %d): %s occurred.  ' + 'Calling MPI_Abort!\n') % (world.rank, error))
            sys.stderr.flush()
            time.sleep(3)
            world.abort(42)
    atexit.register(cleanup)