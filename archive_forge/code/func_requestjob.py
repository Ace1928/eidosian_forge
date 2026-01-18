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
@Pyro4.expose
@Pyro4.oneway
def requestjob(self):
    """Request jobs from the dispatcher, in a perpetual loop until :meth:`~gensim.models.lsi_worker.Worker.getstate`
        is called.

        Raises
        ------
        RuntimeError
            If `self.model` is None (i.e. worker not initialized).

        """
    if self.model is None:
        raise RuntimeError('worker must be initialized before receiving jobs')
    job = None
    while job is None and (not self.finished):
        try:
            job = self.dispatcher.getjob(self.myid)
        except Queue.Empty:
            continue
    if job is not None:
        logger.info('worker #%s received job #%i', self.myid, self.jobsdone)
        self.processjob(job)
        self.dispatcher.jobdone(self.myid)
    else:
        logger.info('worker #%i stopping asking for jobs', self.myid)