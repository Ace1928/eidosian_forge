import copy
from typing import Any, Dict
import stanio
@property
def cmdstan_config(self) -> Dict[str, Any]:
    """
        Returns a dictionary containing a set of name, value pairs
        parsed out of the Stan CSV file header.  These include the
        command configuration and the CSV file header row information.
        Uses deepcopy for immutability.
        """
    return copy.deepcopy(self._cmdstan_config)