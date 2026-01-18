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
def export_to_file(self, file: str) -> bool:
    """Write the model as a protocol buffer to 'file'.

        Args:
          file: file to write the model to. If the filename ends with 'txt', the
            model will be written as a text file, otherwise, the binary format will
            be used.

        Returns:
          True if the model was correctly written.
        """
    return swig_helper.CpSatHelper.write_model_to_file(self.__model, file)