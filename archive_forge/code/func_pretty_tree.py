from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities.power_tools import (
def pretty_tree(self) -> None:
    print(prettydict(self.tree))