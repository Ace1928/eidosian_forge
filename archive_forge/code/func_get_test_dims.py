from io import BytesIO
from itertools import product
import random
from typing import Any, List
import torch
def get_test_dims(min: int, max: int, *, n: int) -> List[int]:
    return [test_dims_rng.randint(min, max) for _ in range(n)]