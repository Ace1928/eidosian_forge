import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Type, Union
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.api import QAM, QuantumExecutable, QAMExecutionResult
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.quil import Program
from pyquil.quilatom import Label, LabelPlaceholder, MemoryReference
from pyquil.quilbase import (
def find_label(self, label: Union[Label, LabelPlaceholder]) -> int:
    """
        Helper function that iterates over the program and looks for a JumpTarget that has a
        Label matching the input label.

        :param label: Label object to search for in program
        :return: Program index where ``label`` is found
        """
    assert self.program is not None
    for index, action in enumerate(self.program):
        if isinstance(action, JumpTarget):
            if label == action.label:
                return index
    raise RuntimeError('Improper program - Jump Target not found in the input program!')