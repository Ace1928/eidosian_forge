from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel
def log_densities(self, observations):
    """
        Compute the log densities of the observations using (a) GMM(s).

        Parameters
        ----------
        observations : numpy array
            Observations (i.e. multi-band spectral flux features).

        Returns
        -------
        numpy array, shape (N, num_gmms)
            Log densities of the observations, the columns represent the
            observation log probability densities for the individual GMMs.

        """
    num_gmms = sum([len(pattern) for pattern in self.pattern_files])
    log_densities = np.empty((len(observations), num_gmms), dtype=np.float)
    i = 0
    for pattern in self.pattern_files:
        for gmm in pattern:
            log_densities[:, i] = gmm.score(observations)
            i += 1
    return log_densities