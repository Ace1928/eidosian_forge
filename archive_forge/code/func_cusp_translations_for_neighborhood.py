from .shapes import compute_hyperbolic_shapes
from .cuspCrossSection import ComplexCuspCrossSection
def cusp_translations_for_neighborhood(neighborhood, verified=False, bits_prec=None):
    manifold = neighborhood.manifold()
    areas = [neighborhood.volume(i) * 2 for i in range(manifold.num_cusps())]
    return cusp_translations_for_manifold(manifold, areas=areas, check_std_form=verified, verified=verified, bits_prec=bits_prec)