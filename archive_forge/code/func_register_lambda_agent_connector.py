from typing import Any, Callable, Type
import numpy as np
import tree  # dm_tree
from ray.rllib.connectors.connector import (
from ray.rllib.connectors.registry import register_connector
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
def register_lambda_agent_connector(name: str, fn: Callable[[Any], Any]) -> Type[AgentConnector]:
    """A util to register any simple transforming function as an AgentConnector

    The only requirement is that fn should take a single data object and return
    a single data object.

    Args:
        name: Name of the resulting actor connector.
        fn: The function that transforms env / agent data.

    Returns:
        A new AgentConnector class that transforms data using fn.
    """

    class LambdaAgentConnector(AgentConnector):

        def transform(self, ac_data: AgentConnectorDataType) -> AgentConnectorDataType:
            return AgentConnectorDataType(ac_data.env_id, ac_data.agent_id, fn(ac_data.data))

        def to_state(self):
            return (name, None)

        @staticmethod
        def from_state(ctx: ConnectorContext, params: Any):
            return LambdaAgentConnector(ctx)
    LambdaAgentConnector.__name__ = name
    LambdaAgentConnector.__qualname__ = name
    register_connector(name, LambdaAgentConnector)
    return LambdaAgentConnector