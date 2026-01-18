import logging
import threading
import time
import numpy as np
@steps_per_execution.setter
def steps_per_execution(self, value):
    self._steps_per_execution.assign(value)
    self.init_spe = value