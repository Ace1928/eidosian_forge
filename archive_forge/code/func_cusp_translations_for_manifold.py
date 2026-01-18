from .shapes import compute_hyperbolic_shapes
from .cuspCrossSection import ComplexCuspCrossSection
def cusp_translations_for_manifold(manifold, verified, areas=None, check_std_form=True, bits_prec=None):
    shapes = compute_hyperbolic_shapes(manifold, verified=verified, bits_prec=bits_prec)
    c = ComplexCuspCrossSection.fromManifoldAndShapes(manifold, shapes)
    if areas:
        RF = shapes[0].real().parent()
        c.normalize_cusps([RF(area) for area in areas])
        if check_std_form:
            c.ensure_std_form()
    else:
        c.ensure_std_form(allow_scaling_up=True)
    c.ensure_disjoint_on_edges()
    return c.all_normalized_translations()