import numpy as np
def leave_index_tree(self, X: np.ndarray) -> np.ndarray:
    """Computes the leave index for all trees."""
    if len(X.shape) == 1:
        X = X.reshape((1, -1))
    outputs = []
    for row in X:
        outs = []
        for tree_id in self.tree_ids:
            outs.append(self.leaf_index_tree(row, tree_id))
        outputs.append(outs)
    return np.array(outputs)