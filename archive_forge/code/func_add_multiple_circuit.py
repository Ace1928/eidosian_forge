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
def add_multiple_circuit(self, arcs: Sequence[ArcT]) -> Constraint:
    """Adds a multiple circuit constraint, aka the 'VRP' constraint.

        The direct graph where arc #i (from tails[i] to head[i]) is present iff
        literals[i] is true must satisfy this set of properties:
        - #incoming arcs == 1 except for node 0.
        - #outgoing arcs == 1 except for node 0.
        - for node zero, #incoming arcs == #outgoing arcs.
        - There are no duplicate arcs.
        - Self-arcs are allowed except for node 0.
        - There is no cycle in this graph, except through node 0.

        Args:
          arcs: a list of arcs. An arc is a tuple (source_node, destination_node,
            literal). The arc is selected in the circuit if the literal is true.
            Both source_node and destination_node must be integers between 0 and the
            number of nodes - 1.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          ValueError: If the list of arcs is empty.
        """
    if not arcs:
        raise ValueError('add_multiple_circuit expects a non-empty array of arcs')
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    for arc in arcs:
        tail = cmh.assert_is_int32(arc[0])
        head = cmh.assert_is_int32(arc[1])
        lit = self.get_or_make_boolean_index(arc[2])
        model_ct.routes.tails.append(tail)
        model_ct.routes.heads.append(head)
        model_ct.routes.literals.append(lit)
    return ct