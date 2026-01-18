import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def override_by(self, other: 'PhasedFSimCharacterization') -> 'PhasedFSimCharacterization':
    """Overrides other parameters that are not None.

        Args:
            other: Parameters to use for override.

        Returns:
            New instance of PhasedFSimCharacterization with values from other if set (values from
            other that are not None). Otherwise the current values are used.
        """
    return other.merge_with(self)