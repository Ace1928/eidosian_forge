import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
def only_enforce_if(self, *boolvar) -> 'Constraint':
    """Adds an enforcement literal to the constraint.

        This method adds one or more literals (that is, a boolean variable or its
        negation) as enforcement literals. The conjunction of all these literals
        determines whether the constraint is active or not. It acts as an
        implication, so if the conjunction is true, it implies that the constraint
        must be enforced. If it is false, then the constraint is ignored.

        BoolOr, BoolAnd, and linear constraints all support enforcement literals.

        Args:
          *boolvar: One or more Boolean literals.

        Returns:
          self.
        """
    for lit in expand_generator_or_tuple(boolvar):
        if cmh.is_boolean(lit) and lit or (isinstance(lit, numbers.Integral) and lit == 1):
            pass
        elif cmh.is_boolean(lit) and (not lit) or (isinstance(lit, numbers.Integral) and lit == 0):
            self.__constraint.enforcement_literal.append(self.__cp_model.new_constant(0).index)
        else:
            self.__constraint.enforcement_literal.append(cast(Union[IntVar, _NotBooleanVariable], lit).index)
    return self