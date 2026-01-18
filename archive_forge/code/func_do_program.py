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
def do_program(self, program: Program) -> 'AbstractQuantumSimulator':
    """
        Perform a sequence of gates contained within a program.

        :param program: The program
        :return: self
        """
    for gate in program:
        if not isinstance(gate, Gate):
            raise ValueError('Can only compute the simulate a program composed of `Gate`s')
        self.do_gate(gate)
    return self