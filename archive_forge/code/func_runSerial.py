import multiprocessing
import os
import re
import sys
import time
from .processes import ForkedProcess
from .remoteproxy import ClosedError
def runSerial(self):
    if self.showProgress:
        self.progressDlg.__enter__()
        self.progressDlg.setMaximum(len(self.tasks))
    self.progress = {os.getpid(): []}
    return Tasker(self, None, self.tasks, self.kwds)