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
def getworkers(self):
    """Get pyro URIs of all registered workers.

        Returns
        -------
        list of URIs
            The pyro URIs for each worker.

        """
    return [worker._pyroUri for worker in self.workers.values()]