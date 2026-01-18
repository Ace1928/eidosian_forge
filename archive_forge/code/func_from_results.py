import warnings
import numpy as np
from . import tools
@classmethod
def from_results(cls, filter_results):
    a = filter_results.initial_state
    Pstar = filter_results.initial_state_cov
    Pinf = filter_results.initial_diffuse_state_cov
    return cls.from_components(filter_results.model.k_states, a=a, Pstar=Pstar, Pinf=Pinf)