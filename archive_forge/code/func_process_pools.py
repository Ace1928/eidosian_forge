import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def process_pools(self):
    if self.g_cons is not None:
        self.process_gpool()
    self.process_fpool()
    self.proc_minimisers()