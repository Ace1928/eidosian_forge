from typing import Any, List, Optional, Union
import numpy as np
def normal_to_integers(value: Any, mean: int, sigma: float, q: int=1) -> Union[int, List[int]]:
    res = normal_to_discrete(value, mean=mean, sigma=sigma, q=q)
    if np.isscalar(res):
        return int(res)
    return [int(x) for x in res]