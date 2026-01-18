from autograd.extend import primitive, defvjp, vspace
from autograd.builtins import tuple
from autograd import make_vjp
def fixed_point_vjp(ans, f, a, x0, distance, tol):

    def rev_iter(params):
        a, x_star, x_star_bar = params
        vjp_x, _ = make_vjp(f(a))(x_star)
        vs = vspace(x_star)
        return lambda g: vs.add(vjp_x(g), x_star_bar)
    vjp_a, _ = make_vjp(lambda x, y: f(x)(y))(a, ans)
    return lambda g: vjp_a(fixed_point(rev_iter, tuple((a, ans, g)), vspace(x0).zeros(), distance, tol))