import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
def test_issue_142(self):
    CHOICES = [((1, 2, 3), 4, 3), ((1, 2, 2), 4, 3), ((1, 3, 3), 4, 3)]
    try:
        _oldFlatten = normalize_index.flatten
        normalize_index.flatten = False
        m = ConcreteModel()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.CHOICES = Set(initialize=CHOICES, dimen=3)
            self.assertIn('Ignoring non-None dimen (3) for set CHOICES', output.getvalue())
        self.assertEqual(m.CHOICES.dimen, None)
        m.x = Var(m.CHOICES)

        def c_rule(m, a, b, c):
            return m.x[a, b, c] == 0
        m.c = Constraint(m.CHOICES, rule=c_rule)
        output = StringIO()
        m.CHOICES.pprint(ostream=output)
        m.x.pprint(ostream=output)
        m.c.pprint(ostream=output)
        ref = '\nCHOICES : Size=1, Index=None, Ordered=Insertion\n    Key  : Dimen : Domain : Size : Members\n    None :  None :    Any :    3 : {((1, 2, 3), 4, 3), ((1, 2, 2), 4, 3), ((1, 3, 3), 4, 3)}\nx : Size=3, Index=CHOICES\n    Key               : Lower : Value : Upper : Fixed : Stale : Domain\n    ((1, 2, 2), 4, 3) :  None :  None :  None : False :  True :  Reals\n    ((1, 2, 3), 4, 3) :  None :  None :  None : False :  True :  Reals\n    ((1, 3, 3), 4, 3) :  None :  None :  None : False :  True :  Reals\nc : Size=3, Index=CHOICES, Active=True\n    Key               : Lower : Body           : Upper : Active\n    ((1, 2, 2), 4, 3) :   0.0 : x[(1,2,2),4,3] :   0.0 :   True\n    ((1, 2, 3), 4, 3) :   0.0 : x[(1,2,3),4,3] :   0.0 :   True\n    ((1, 3, 3), 4, 3) :   0.0 : x[(1,3,3),4,3] :   0.0 :   True\n'.strip()
        self.assertEqual(output.getvalue().strip(), ref)
        normalize_index.flatten = True
        m = ConcreteModel()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.CHOICES = Set(initialize=CHOICES)
            self.assertEqual('', output.getvalue())
        self.assertEqual(m.CHOICES.dimen, 5)
        m.x = Var(m.CHOICES)

        def c_rule(m, a1, a2, a3, b, c):
            return m.x[a1, a2, a3, b, c] == 0
        m.c = Constraint(m.CHOICES, rule=c_rule)
        output = StringIO()
        m.CHOICES.pprint(ostream=output)
        m.x.pprint(ostream=output)
        m.c.pprint(ostream=output)
        ref = '\nCHOICES : Size=1, Index=None, Ordered=Insertion\n    Key  : Dimen : Domain : Size : Members\n    None :     5 :    Any :    3 : {(1, 2, 3, 4, 3), (1, 2, 2, 4, 3), (1, 3, 3, 4, 3)}\nx : Size=3, Index=CHOICES\n    Key             : Lower : Value : Upper : Fixed : Stale : Domain\n    (1, 2, 2, 4, 3) :  None :  None :  None : False :  True :  Reals\n    (1, 2, 3, 4, 3) :  None :  None :  None : False :  True :  Reals\n    (1, 3, 3, 4, 3) :  None :  None :  None : False :  True :  Reals\nc : Size=3, Index=CHOICES, Active=True\n    Key             : Lower : Body         : Upper : Active\n    (1, 2, 2, 4, 3) :   0.0 : x[1,2,2,4,3] :   0.0 :   True\n    (1, 2, 3, 4, 3) :   0.0 : x[1,2,3,4,3] :   0.0 :   True\n    (1, 3, 3, 4, 3) :   0.0 : x[1,3,3,4,3] :   0.0 :   True\n'.strip()
        self.assertEqual(output.getvalue().strip(), ref)
    finally:
        normalize_index.flatten = _oldFlatten