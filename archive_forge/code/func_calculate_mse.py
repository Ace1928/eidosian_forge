import numpy as np
import scipy
import sklearn
import meld
import graphtools as gt
def calculate_mse(self, estimate):
    """Calculated MSE between the ground truth PDF and an estimate"""
    return sklearn.metrics.mean_squared_error(self.pdf, estimate)