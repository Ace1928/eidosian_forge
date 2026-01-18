import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def construct_types(self):
    types = [(int, 'integer'), (str, 'yfiles'), (str, 'string'), (int, 'int'), (int, 'long'), (float, 'float'), (float, 'double'), (bool, 'boolean')]
    try:
        import numpy as np
    except:
        pass
    else:
        types = [(np.float64, 'float'), (np.float32, 'float'), (np.float16, 'float'), (np.int_, 'int'), (np.int8, 'int'), (np.int16, 'int'), (np.int32, 'int'), (np.int64, 'int'), (np.uint8, 'int'), (np.uint16, 'int'), (np.uint32, 'int'), (np.uint64, 'int'), (np.int_, 'int'), (np.intc, 'int'), (np.intp, 'int')] + types
    self.xml_type = dict(types)
    self.python_type = dict((reversed(a) for a in types))