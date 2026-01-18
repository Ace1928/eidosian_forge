import copy
from math import sqrt, log, exp
from itertools import cycle
import warnings
import numpy
from . import tools
def computeParams(self, params):
    """Computes the parameters depending on :math:`\\lambda`. It needs to
        be called again if :math:`\\lambda` changes during evolution.

        :param params: A dictionary of the manually set parameters.
        """
    self.lambda_ = params.get('lambda_', 1)
    self.d = params.get('d', 1.0 + self.dim / (2.0 * self.lambda_))
    self.ptarg = params.get('ptarg', 1.0 / (5 + sqrt(self.lambda_) / 2.0))
    self.cp = params.get('cp', self.ptarg * self.lambda_ / (2 + self.ptarg * self.lambda_))
    self.cc = params.get('cc', 2.0 / (self.dim + 2.0))
    self.ccov = params.get('ccov', 2.0 / (self.dim ** 2 + 6.0))
    self.pthresh = params.get('pthresh', 0.44)