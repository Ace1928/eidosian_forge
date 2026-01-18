import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def regularize_tensor(metric_tensor):
    tensor_reg = np.real(sqrtm(np.matmul(metric_tensor, metric_tensor)))
    return (tensor_reg + self.reg * np.identity(metric_tensor.shape[0])) / (1 + self.reg)