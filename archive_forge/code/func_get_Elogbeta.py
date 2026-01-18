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
def get_Elogbeta(self):
    """Get the log (posterior) probabilities for each topic.

        Returns
        -------
        numpy.ndarray
            Posterior probabilities for each topic.
        """
    return dirichlet_expectation(self.get_lambda())