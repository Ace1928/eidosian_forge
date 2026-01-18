from cvxpy.atoms.quad_form import QuadForm, SymbolicQuadForm
from cvxpy.expressions.variable import Variable
def restore_quad_forms(expr, quad_forms) -> None:
    for idx, arg in enumerate(expr.args):
        if isinstance(arg, Variable) and arg.id in quad_forms:
            expr.args[idx] = quad_forms[arg.id][2]
        else:
            restore_quad_forms(arg, quad_forms)