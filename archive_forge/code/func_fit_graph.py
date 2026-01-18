import numpy as np
import scipy
import sklearn
import meld
import graphtools as gt
def fit_graph(self, data, n_pca=100, **kwargs):
    """Fits a graphtools.Graph to input data

        Parameters
        ----------
        data : array, shape=[n_samples, n_observations]
            Input data
        **kwargs : dict
            Keyword arguments passed to gt.Graph()

        Returns
        -------
        graph : graphtools.Graph
            Graph fit to data

        """
    self.graph = gt.Graph(data, n_pca=n_pca, use_pygsp=True, random_state=self.seed, **kwargs)
    return self.graph