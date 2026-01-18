from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union, TYPE_CHECKING
import re
import numpy as np
from cirq import ops, linalg, protocols, value
def is_valid_qasm_id(self, id_str: str) -> bool:
    """Test if id_str is a valid id in QASM grammar."""
    return self.valid_id_re.match(id_str) is not None