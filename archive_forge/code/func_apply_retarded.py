import numpy as np
def apply_retarded(self, energy, X):
    """Apply retarded Green function to X.
        
        Returns the matrix product G^r(e) . X
        """
    return np.linalg.solve(self.retarded(energy, inverse=True), X)