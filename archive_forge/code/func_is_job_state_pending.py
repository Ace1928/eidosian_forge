import os
import pwd
import re
import subprocess
import time
import xml.dom.minidom
import random
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
def is_job_state_pending(self):
    """Return True, unless job is in the "zombie" status"""
    time_diff = time.time() - self._job_info_creation_time
    if self.is_zombie():
        sge_debug_print("DONE! QJobInfo.IsPending found in 'zombie' list, returning False so claiming done!\n{0}".format(self))
        is_pending_status = False
    elif self.is_initializing() and time_diff > 600:
        sge_debug_print("FAILURE! QJobInfo.IsPending found long running at {1} seconds'initializing' returning False for to break loop!\n{0}".format(self, time_diff))
        is_pending_status = True
    else:
        is_pending_status = True
    return is_pending_status