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
def do_mstep(self, rho, other, extra_pass=False):
    """Maximization step: use linear interpolation between the existing topics and
        collected sufficient statistics in `other` to update the topics.

        Parameters
        ----------
        rho : float
            Learning rate.
        other : :class:`~gensim.models.ldamodel.LdaModel`
            The model whose sufficient statistics will be used to update the topics.
        extra_pass : bool, optional
            Whether this step required an additional pass over the corpus.

        """
    logger.debug('updating topics')
    previous_Elogbeta = self.state.get_Elogbeta()
    self.state.blend(rho, other)
    current_Elogbeta = self.state.get_Elogbeta()
    self.sync_state(current_Elogbeta)
    self.print_topics(5)
    diff = mean_absolute_difference(previous_Elogbeta.ravel(), current_Elogbeta.ravel())
    logger.info('topic diff=%f, rho=%f', diff, rho)
    if self.optimize_eta:
        self.update_eta(self.state.get_lambda(), rho)
    if not extra_pass:
        self.num_updates += other.numdocs