from typing import List
from .plan import Plan
@property
def rules(self) -> List[Rule]:
    """List of predefined rules for this optimizer."""
    raise NotImplementedError