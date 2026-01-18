from collections import defaultdict
from typing import Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.nonpos import NonNeg, NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.cvxcore.python import canonInterface
def lower_ineq_to_nonneg(inequality):
    lhs = inequality.args[0]
    rhs = inequality.args[1]
    return NonNeg(rhs - lhs, constr_id=inequality.constr_id)