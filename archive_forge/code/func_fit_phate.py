import numpy as np
import scipy
import sklearn
import meld
import graphtools as gt
def fit_phate(self, data, **kwargs):
    """Generates a 3D phate embedding of input data

        Parameters
        ----------
        data : array, shape=[n_samples, n_observations]
            Description of parameter `data`.
        **kwargs : dict
            Keyword arguments passed to phate.PHATE().

        Returns
        -------
        data_phate : array, shape=[n_samples, 3]
            Normalized PHATE embedding for input data.

        """
    import phate
    self.set_phate(phate.PHATE(n_components=3, **kwargs).fit_transform(data))
    return self.data_phate