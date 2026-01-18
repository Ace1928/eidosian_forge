from pythran.analyses import Identifiers
from pythran.passmanager import Transformation
import gast as ast
from functools import reduce
from collections import OrderedDict
from copy import deepcopy
def get_new_id(self):
    i = 0
    while 1:
        new_id = '{}{}'.format(NormalizeTuples.tuple_name, i)
        if new_id not in self.ids:
            self.ids.add(new_id)
            return new_id
        else:
            i += 1