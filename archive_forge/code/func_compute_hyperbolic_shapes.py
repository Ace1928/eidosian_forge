from . import verifyHyperbolicity
def compute_hyperbolic_shapes(manifold, verified, bits_prec=None):
    shapes = manifold.tetrahedra_shapes('rect', intervals=verified, bits_prec=bits_prec)
    if verified:
        verifyHyperbolicity.check_logarithmic_gluing_equations_and_positively_oriented_tets(manifold, shapes)
    else:
        sol_type = manifold.solution_type()
        if not sol_type == 'all tetrahedra positively oriented':
            raise RuntimeError("Manifold has non-geometric solution type '%s'." % sol_type)
    return shapes