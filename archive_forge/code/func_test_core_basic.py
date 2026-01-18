import inspect
import copy
import pickle
from sympy.physics.units import meter
from sympy.testing.pytest import XFAIL, raises
from sympy.core.basic import Atom, Basic
from sympy.core.singleton import SingletonRegistry
from sympy.core.symbol import Str, Dummy, Symbol, Wild
from sympy.core.numbers import (E, I, pi, oo, zoo, nan, Integer,
from sympy.core.relational import (Equality, GreaterThan, LessThan, Relational,
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import Derivative, Function, FunctionClass, Lambda, \
from sympy.sets.sets import Interval
from sympy.core.multidimensional import vectorize
from sympy.external.gmpy import HAS_GMPY
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.external import import_module
from sympy.functions import (Piecewise, lowergamma, acosh, chebyshevu,
from sympy.geometry.entity import GeometryEntity
from sympy.geometry.point import Point
from sympy.geometry.ellipse import Circle, Ellipse
from sympy.geometry.line import Line, LinearEntity, Ray, Segment
from sympy.geometry.polygon import Polygon, RegularPolygon, Triangle
from sympy.integrals.integrals import Integral
from sympy.core.logic import Logic
from sympy.matrices import Matrix, SparseMatrix
from sympy.ntheory.generate import Sieve
from sympy.physics.paulialgebra import Pauli
from sympy.physics.units import Unit
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.orderings import lex
from sympy.polys.polytools import Poly
from sympy.printing.latex import LatexPrinter
from sympy.printing.mathml import MathMLContentPrinter, MathMLPresentationPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.printer import Printer
from sympy.printing.python import PythonPrinter
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
def test_core_basic():
    for c in (Atom, Atom(), Basic, Basic(), SingletonRegistry, S):
        check(c)