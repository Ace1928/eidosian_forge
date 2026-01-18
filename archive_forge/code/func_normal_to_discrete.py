from typing import Any, List, Optional, Union
import numpy as np
def normal_to_discrete(value: Any, mean: float, sigma: float, q: float) -> Any:
    return np.round(value * sigma / q) * q + mean