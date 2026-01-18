from typing import Dict, List, NoReturn, Sequence, Union
from torchgen.api.types import (
def direct_solve(goal: NamedCType) -> str:
    return solve(goal, direct=True)