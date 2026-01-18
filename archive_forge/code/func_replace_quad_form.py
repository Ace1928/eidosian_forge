from cvxpy.atoms.quad_form import QuadForm, SymbolicQuadForm
from cvxpy.expressions.variable import Variable
def replace_quad_form(expr, idx, quad_forms):
    quad_form = expr.args[idx]
    placeholder = Variable(quad_form.shape, var_id=quad_form.id)
    expr.args[idx] = placeholder
    quad_forms[placeholder.id] = (expr, idx, quad_form)
    return quad_forms