import os
import math
import dill as pickle
def test_c2adder():
    pc2adder = pickle.dumps(c2adder)
    pc2add5 = pickle.loads(pc2adder)(x)
    assert pc2add5(y) == x + y