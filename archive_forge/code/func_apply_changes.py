from ray.rllib.connectors.connector import (
from ray.util.annotations import PublicAPI
from ray.rllib.utils.filter import Filter
def apply_changes(self, other: 'Filter', *args, **kwargs) -> None:
    """Updates self with state from other filter."""
    return self.filter.apply_changes(other, *args, **kwargs)