import random
from ._fold_storage import FoldStorage
from ._fold_storage import _FoldFile
def create_fold_sets(self, fold_size, folds_count):
    """Create all folds for all permutations."""
    folds = []
    passed_folds_count = 0
    while passed_folds_count < folds_count:
        folds.append(self._make_learn_folds(fold_size, folds_count - passed_folds_count))
        current_learn_folds = folds[-1]
        passed_folds_count += len(current_learn_folds)
    return folds