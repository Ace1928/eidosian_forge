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
@Pyro4.oneway
@utils.synchronous('lock_update')
def jobdone(self, workerid):
    """A worker has finished its job. Log this event and then asynchronously transfer control back to the worker.

        Callback used by workers to notify when their job is done.

        The job done event is logged and then control is asynchronously transfered back to the worker
        (who can then request another job). In this way, control flow basically oscillates between
        :meth:`gensim.models.lsi_dispatcher.Dispatcher.jobdone` and :meth:`gensim.models.lsi_worker.Worker.requestjob`.

        Parameters
        ----------
        workerid : int
            The ID of the worker that finished the job (used for logging).

        """
    self._jobsdone += 1
    logger.info('worker #%s finished job #%i', workerid, self._jobsdone)
    worker = self.workers[workerid]
    worker.requestjob()