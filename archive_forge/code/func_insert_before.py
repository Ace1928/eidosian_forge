import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import gymnasium as gym
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def insert_before(self, name: str, connector: Connector):
    """Insert a new connector before connector <name>

        Args:
            name: name of the connector before which a new connector
                will get inserted.
            connector: a new connector to be inserted.
        """
    idx = -1
    for idx, c in enumerate(self.connectors):
        if c.__class__.__name__ == name:
            break
    if idx < 0:
        raise ValueError(f'Can not find connector {name}')
    self.connectors.insert(idx, connector)
    logger.info(f'Inserted {connector.__class__.__name__} before {name} to {self.__class__.__name__}.')