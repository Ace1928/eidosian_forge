import collections
import os
import sys
import queue
import subprocess
import traceback
import weakref
from functools import partial
from threading import Thread
from jedi._compatibility import pickle_dump, pickle_load
from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.access import DirectObjectAccess, AccessPath, \
from jedi.api.exceptions import InternalError
def delete_inference_state(self, inference_state_id):
    """
        Currently we are not deleting inference_state instantly. They only get
        deleted once the subprocess is used again. It would probably a better
        solution to move all of this into a thread. However, the memory usage
        of a single inference_state shouldn't be that high.
        """
    self._inference_state_deletion_queue.append(inference_state_id)