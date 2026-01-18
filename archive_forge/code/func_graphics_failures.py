import sys
import getopt
import snappy
import snappy.snap.test
import spherogram.test
import snappy.matrix
import snappy.verify.test
import snappy.ptolemy.test
import snappy.raytracing.cohomology_fractal
import snappy.raytracing.geodesic_tube_info
import snappy.raytracing.geodesics
import snappy.raytracing.ideal_raytracing_data
import snappy.raytracing.upper_halfspace_utilities
import snappy.drilling
import snappy.exterior_to_link.test
import snappy.pari
from snappy.sage_helper import (_within_sage, doctest_modules, cyopengl_works,
from snappy import numeric_output_checker
def graphics_failures(verbose, windows, use_modernopengl):
    if cyopengl_works():
        print('Testing graphics ...')
        import snappy.CyOpenGL
        result = doctest_modules([snappy.CyOpenGL], verbose=verbose).failed
        snappy.Manifold('m004').dirichlet_domain().view().test()
        snappy.Manifold('m125').cusp_neighborhood().view().test()
        if use_modernopengl:
            snappy.Manifold('m004').inside_view().test()
        snappy.Manifold('4_1').browse().test()
        snappy.ManifoldHP('m004').dirichlet_domain().view().test()
        snappy.ManifoldHP('m125').cusp_neighborhood().view().test()
        if use_modernopengl:
            snappy.ManifoldHP('m004').inside_view().test()
        if root_is_fake():
            root = tk_root()
            if root:
                if windows:
                    print('Close the root window to finish.')
                else:
                    print('The windows will close in a few seconds.\nSpecify -w or --windows to avoid this.')
                    root.after(7000, root.destroy)
                root.mainloop()
    else:
        print('***Warning***: CyOpenGL not installed, so not tested')
        result = 0
    return result