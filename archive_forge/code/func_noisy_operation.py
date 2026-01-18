import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
def noisy_operation(self, operation):
    yield operation
    if cirq.LineQubit(0) in operation.qubits:
        yield cirq.H(cirq.LineQubit(0))