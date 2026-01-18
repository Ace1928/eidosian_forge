from parlai.core.teachers import FixedDialogTeacher
from .build import build
import json
import os
def list_to_str(lst):
    s = ''
    for ele in lst:
        if s:
            s += ' ' + ele
        else:
            s = ele
    return s