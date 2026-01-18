from parlai.core.teachers import FbDialogTeacher
from parlai.utils.misc import warn_once
from .build import build
from parlai.utils.strings import normalize_reply
import parlai.utils.logging as logging
import copy
import os
def normalize_replies(self, x):
    xs = x.split('\n')
    xs2 = []
    for x in xs:
        if 'your persona:' in x:
            x = x[len('your persona: '):]
            x = normalize_reply(x)
            x = 'your persona: ' + x
        else:
            x = normalize_reply(x)
        xs2.append(x)
    return '\n'.join(xs2)