from typing import Any, Callable, List, Optional, Tuple
@property
def num_results(self) -> int:
    """Number of received (successful) results."""
    return len(self._results)