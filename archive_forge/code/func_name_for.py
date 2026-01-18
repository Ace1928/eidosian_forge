import time
from collections import defaultdict
from functools import partial
from typing import DefaultDict
import torch
def name_for(node):
    kind = node.kind()[node.kind().index('::') + 2:]
    op_id_counter[kind] += 1
    return (kind, name_prefix + kind + '_' + str(op_id_counter[kind]))