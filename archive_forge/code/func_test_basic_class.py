import os
import math
import dill as pickle
def test_basic_class():
    pbasic2 = pickle.dumps(basic2)
    _pbasic2 = pickle.loads(pbasic2)()
    pbasic = pickle.dumps(basic)
    _pbasic = pickle.loads(pbasic)()