import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
def convert_to_memory_friendly(self):
    """Remove the stored gensim models and only keep their ttdas.

        This frees up memory, but you won't have access to the individual  models anymore if you intended to use them
        outside of the ensemble.
        """
    self.tms = []
    self.memory_friendly_ttda = True