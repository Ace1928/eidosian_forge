from typing import Any, List, Optional, Union
import numpy as np
def normal_to_continuous(value: Any, mean: float, sigma: float) -> Any:
    return value * sigma + mean