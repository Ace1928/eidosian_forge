import datetime
from typing import Mapping
import pandas as pd
import cirq
import cirq_google as cg
import numpy as np
@property
def measurements(self) -> Mapping[str, np.ndarray]:
    return {'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])}