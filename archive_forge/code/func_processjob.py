import os
import sys
import logging
import argparse
import threading
import tempfile
import queue as Queue
import Pyro4
from gensim.models import lsimodel
from gensim import utils
@utils.synchronous('lock_update')
def processjob(self, job):
    """Incrementally process the job and potentially logs progress.

        Parameters
        ----------
        job : iterable of list of (int, float)
            Corpus in BoW format.

        """
    self.model.add_documents(job)
    self.jobsdone += 1
    if SAVE_DEBUG and self.jobsdone % SAVE_DEBUG == 0:
        fname = os.path.join(tempfile.gettempdir(), 'lsi_worker.pkl')
        self.model.save(fname)