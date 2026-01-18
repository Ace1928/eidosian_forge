import random
from ._fold_storage import FoldStorage
from ._fold_storage import _FoldFile
def fold_groups_files_generator(self, folds_groups, fold_offset):
    """Create folds storages for all folds in folds_groups. Generator."""
    fold_num = 0
    for fold_group in folds_groups:
        learn_folds = []
        skipped_folds = []
        for learn_set in fold_group:
            fold_num += 1
            if fold_offset < fold_num:
                fold_file = self.create_fold(learn_set, 'fold', fold_num)
                learn_folds.append(fold_file)
            elif fold_offset >= fold_num:
                fold_file = self.create_fold(learn_set, 'offset{}_skipped'.format(fold_offset), fold_num)
                skipped_folds.append(fold_file)
        rest_folds = self._write_folds(learn_folds + skipped_folds, fold_num, fold_offset)
        yield (learn_folds, skipped_folds, rest_folds)