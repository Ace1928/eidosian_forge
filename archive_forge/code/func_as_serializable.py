from ray.rllib.connectors.connector import (
from ray.util.annotations import PublicAPI
from ray.rllib.utils.filter import Filter
def as_serializable(self) -> 'Filter':
    return self.filter.as_serializable()