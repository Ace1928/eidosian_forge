from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def ptolemy_generalized_obstruction_classes(self, N):
    """
        Return the obstruction classes needed to compute
        PGL(N,C)-representations for any N.

        It returns a list with
        a representative cocycle for each element in
        ``H^2(M, boundary M; Z/N) / (Z/N)^*``
        where ``(Z/N)^*`` are the units in Z/N.
        The first element in the list always corresponds to the trivial
        obstruction class.
        The generalized ptolemy obstruction classes are thus a generalization
        of the ptolemy obstruction classes that allow to find all
        boundary-unipotent
        PGL(N,C)-representations including those that do not lift to
        boundary-unipotent SL(N,C)-representations for N odd or
        SL(N,C)/{+1,-1}-representations for N even.

        For example, the figure eight not complement has three obstruction
        classes for N = 4 up to equivalence:

        >>> from regina import NExampleTriangulation
        >>> N = NTriangulationForPtolemy(NExampleTriangulation.figureEightKnotComplement())
        >>> c = N.ptolemy_generalized_obstruction_classes(4)
        >>> len(c)
        3

        See  help(Manifold.ptolemy_generalized_obstruction_classes()) for more.
        """
    return manifoldMethods.get_generalized_ptolemy_obstruction_classes(self, N)