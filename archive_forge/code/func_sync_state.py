import logging
import numbers
import os
import time
from collections import defaultdict
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from gensim import interfaces, utils, matutils
from gensim.matutils import (
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback
def sync_state(self, current_Elogbeta=None):
    """Propagate the states topic probabilities to the inner object's attribute.

        Parameters
        ----------
        current_Elogbeta: numpy.ndarray
            Posterior probabilities for each topic, optional.
            If omitted, it will get Elogbeta from state.

        """
    if current_Elogbeta is None:
        current_Elogbeta = self.state.get_Elogbeta()
    self.expElogbeta = np.exp(current_Elogbeta)
    assert self.expElogbeta.dtype == self.dtype