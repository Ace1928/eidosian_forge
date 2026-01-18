import abc
from cvxpy.constraints.constraint import Constraint
@property
def dual_residual(self) -> float:
    """Computes the residual (see Constraint.violation for a
        more formal definition) for the dual cone of the current instance
        of `Cone` w.r.t. the recovered dual variables

        Primarily intended to be used for KKT checks
        """
    return self._dual_cone(*self.dual_variables).residual