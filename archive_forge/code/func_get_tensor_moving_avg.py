import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def get_tensor_moving_avg(metric_tensor):
    if self.metric_tensor is None:
        self.metric_tensor = np.identity(metric_tensor.shape[0])
    return self.k / (self.k + 1) * self.metric_tensor + 1 / (self.k + 1) * metric_tensor