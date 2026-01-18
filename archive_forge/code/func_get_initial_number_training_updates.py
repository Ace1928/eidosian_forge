from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
def get_initial_number_training_updates(self):
    return self._number_training_updates