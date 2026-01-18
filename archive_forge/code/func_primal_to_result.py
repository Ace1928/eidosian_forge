import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.error import DCPError
from cvxpy.expressions.expression import Expression
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.utilities import scopes
@staticmethod
def primal_to_result(result):
    """The value of the objective given the solver primal value.
        """
    return -result