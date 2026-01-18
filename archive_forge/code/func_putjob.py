import os
import sys
import logging
import argparse
import threading
import time
from queue import Queue
import Pyro4
from gensim import utils
@Pyro4.expose
def putjob(self, job):
    """Atomically add a job to the queue.

        Parameters
        ----------
        job : iterable of list of (int, float)
            The corpus in BoW format.

        """
    self._jobsreceived += 1
    self.jobs.put(job, block=True, timeout=HUGE_TIMEOUT)
    logger.info('added a new job (len(queue)=%i items)', self.jobs.qsize())