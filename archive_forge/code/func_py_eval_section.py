from . import matrix
from . import homology
from .polynomial import Polynomial
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase, processFileDispatch, processMagmaFile
from . import utilities
from string import Template
import signal
import re
import os
import sys
from urllib.request import Request, urlopen
from urllib.request import quote as urlquote
from urllib.error import HTTPError
def py_eval_section(self):
    """
        Returns a string that can be evaluated in python and contains extra
        information needed to recover solutions from a simplified Ptolemy
        variety.

        >>> from snappy import Manifold, pari
        >>> M = Manifold('4_1')
        >>> p = M.ptolemy_variety(2, obstruction_class = 1)

        Get extra information

        >>> eval_section = p.py_eval_section()
        >>> print(eval_section)    #doctest: +ELLIPSIS
        {'variable_dict' :
             (lambda d: {
            ...

        Turn it into a python object by evaluation.

        >>> obj = eval(eval_section)

        Access the function that expands a solution to the simplified
        Ptolemy variety to a full solution.

        >>> variable_dict = obj['variable_dict']

        Setup a solution and expand it to a full solution, '1' must map to 1

        >>> simplified_solution = {'c_0101_0' : pari('0.5 - 0.866025403784439*I'), '1' : pari(1), 'c_0011_0' : pari(1)}
        >>> full_solution = variable_dict(simplified_solution)

        Full solution is a dictionary with a key for every Ptolemy coordinate

        >>> full_solution['c_1010_1']
        1
        >>> for tet in range(2):
        ...     for i in utilities.quadruples_with_fixed_sum_iterator(2, skipVertices = True):
        ...         c = "c_%d%d%d%d" % i + "_%d" % tet
        ...         assert c in full_solution
        """
    result = '{'
    result += "'variable_dict' :\n     %s" % self.py_eval_variable_dict()
    if isinstance(self._obstruction_class, PtolemyGeneralizedObstructionClass):
        if self._obstruction_class._is_non_trivial(self._N):
            result += ",\n 'non_trivial_generalized_obstruction_class' : True"
    result += '}'
    return result