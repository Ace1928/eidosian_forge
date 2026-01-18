import numpy as np
import scipy
import sklearn
import meld
import graphtools as gt
def generate_ground_truth_pdf(self, data_phate=None):
    """Creates a random density function over input data.

        Takes a set of PHATE coordinates over a set of points and creates an underlying
        ground truth pdf over the points as a convex combination of the input phate
        coordinates.

        Parameters
        ----------
        data_phate : array, shape=[n_samples, 3]
            PHATE embedding for input data.

        Returns
        -------
        pdf
            Ground truth conditional probability of the sample given the data.

        """
    np.random.seed(self.seed)
    if data_phate is not None:
        self.set_phate(data_phate)
    elif self.data_phate is None:
        raise ValueError('data_phate must be set prior to running generate_ground_truth_pdf().')
    data_simplex = np.sort(np.random.uniform(size=2))
    data_simplex = np.hstack([0, data_simplex, 1])
    data_simplex = np.diff(data_simplex)
    np.random.shuffle(data_simplex)
    sort_axis = np.sum(self.data_phate * data_simplex, axis=1)
    self.pdf = scipy.special.expit(sort_axis)
    return self.pdf