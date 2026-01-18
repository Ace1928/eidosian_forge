import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def keep_func(op: 'cirq.Operation'):
    return not isinstance(op.gate, (cirq.SwapPowGate, cirq.XPowGate))