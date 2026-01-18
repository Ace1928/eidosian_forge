import random
from ._fold_storage import FoldStorage
from ._fold_storage import _FoldFile
def clean_folds(self):
    for file in self._folds_storage:
        file.delete()