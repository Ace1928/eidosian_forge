from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build
import copy
import os
def load_cands(self, path):
    return mod_labels(super().load_cands(path), self.task_num)