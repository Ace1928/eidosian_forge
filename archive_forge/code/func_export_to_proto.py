import abc
import dataclasses
import math
import numbers
import typing
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy import typing as npt
import pandas as pd
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.linear_solver.python import model_builder_numbers as mbn
def export_to_proto(self) -> linear_solver_pb2.MPModelProto:
    """Exports the optimization model to a ProtoBuf format."""
    return mbh.to_mpmodel_proto(self.__helper)