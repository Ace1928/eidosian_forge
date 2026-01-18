import numpy as np
import scipy
import sklearn
import meld
import graphtools as gt
def calculate_MELD_likelihood(self, data=None, **kwargs):
    np.random.seed(self.seed)
    if not self.graph:
        if data is not None:
            self.fit_graph(data)
        else:
            raise NameError('Must pass `data` unless graph has already been fit')
    self.meld_op = meld.MELD(**kwargs, verbose=False).fit(self.graph)
    self.sample_densities = self.meld_op.transform(self.sample_labels)
    self.sample_likelihoods = meld.utils.normalize_densities(self.sample_densities)
    self.expt_likelihood = self.sample_likelihoods['expt'].values
    return self.expt_likelihood