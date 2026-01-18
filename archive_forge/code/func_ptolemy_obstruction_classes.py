from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def ptolemy_obstruction_classes(self):
    """
        Returns the obstruction classes needed to compute the
        boundary-unipotent pSL(N,C) = SL(N,C) / {+1, -1} representations for
        even N.
        This is a list of a representative cocycle for class in
        H^2(M, boundary M; Z/2). The first element in the list is always
        representing the trivial obstruction class.

        For example, the figure eight knot complement has two obstruction
        classes:

        >>> from regina import NExampleTriangulation
        >>> N = NTriangulationForPtolemy(NExampleTriangulation.figureEightKnotComplement())
        >>> c = N.ptolemy_obstruction_classes()
        >>> len(c)
        2

        See  help(Manifold.ptolemy_obstruction_classes()) for more.
        """
    return manifoldMethods.get_ptolemy_obstruction_classes(self)