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
def getjob(self, worker_id):
    """Atomically pop a job from the queue.

        Parameters
        ----------
        worker_id : int
            The worker that requested the job.

        Returns
        -------
        iterable of iterable of (int, float)
            The corpus in BoW format.

        """
    logger.info('worker #%i requesting a new job', worker_id)
    job = self.jobs.get(block=True, timeout=1)
    logger.info('worker #%i got a new job (%i left)', worker_id, self.jobs.qsize())
    return job