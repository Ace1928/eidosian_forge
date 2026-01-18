import pickle
import pytest
from sklearn.utils.metaestimators import available_if
@available_if(lambda est: est.available)
def available_func(self):
    """This is a mock available_if function"""
    return self.return_value