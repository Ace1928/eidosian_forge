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
@property
def engine_calibration(self) -> Optional[Calibration]:
    """The underlying device calibration that was used for this user-specific calibration.

        This is a cached property that triggers a network call at the first use.
        """
    if self._calibration is None and self.engine_job is not None:
        self._calibration = self.engine_job.get_calibration()
    return self._calibration