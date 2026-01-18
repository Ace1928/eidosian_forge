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
def engine_job(self) -> Optional[EngineJob]:
    """The cirq_google.EngineJob associated with this calibration request.

        Available only when project_id, program_id and job_id attributes are present.
        """
    if self._engine_job is None and self.project_id and self.program_id and self.job_id:
        engine = Engine(project_id=self.project_id)
        self._engine_job = engine.get_program(self.program_id).get_job(self.job_id)
    return self._engine_job