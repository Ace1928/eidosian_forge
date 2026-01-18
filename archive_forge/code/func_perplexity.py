from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from joblib import effective_n_jobs
from scipy.special import gammaln, logsumexp
from ..base import (
from ..utils import check_random_state, gen_batches, gen_even_slices
from ..utils._param_validation import Interval, StrOptions
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._online_lda_fast import (
from ._online_lda_fast import (
from ._online_lda_fast import (
def perplexity(self, X, sub_sampling=False):
    """Calculate approximate perplexity for data X.

        Perplexity is defined as exp(-1. * log-likelihood per word)

        .. versionchanged:: 0.19
           *doc_topic_distr* argument has been deprecated and is ignored
           because user no longer has access to unnormalized distribution

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        sub_sampling : bool
            Do sub-sampling or not.

        Returns
        -------
        score : float
            Perplexity score.
        """
    check_is_fitted(self)
    X = self._check_non_neg_array(X, reset_n_features=True, whom='LatentDirichletAllocation.perplexity')
    return self._perplexity_precomp_distr(X, sub_sampling=sub_sampling)