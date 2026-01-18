import logging
from typing import Any, Tuple, TYPE_CHECKING
from ray.rllib.connectors.action.clip import ClipActionsConnector
from ray.rllib.connectors.action.immutable import ImmutableActionsConnector
from ray.rllib.connectors.action.lambdas import ConvertToNumpyConnector
from ray.rllib.connectors.action.normalize import NormalizeActionsConnector
from ray.rllib.connectors.action.pipeline import ActionConnectorPipeline
from ray.rllib.connectors.agent.clip_reward import ClipRewardAgentConnector
from ray.rllib.connectors.agent.obs_preproc import ObsPreprocessorConnector
from ray.rllib.connectors.agent.pipeline import AgentConnectorPipeline
from ray.rllib.connectors.agent.state_buffer import StateBufferConnector
from ray.rllib.connectors.agent.view_requirement import ViewRequirementAgentConnector
from ray.rllib.connectors.connector import Connector, ConnectorContext
from ray.rllib.connectors.registry import get_connector
from ray.rllib.connectors.agent.mean_std_filter import (
from ray.util.annotations import PublicAPI, DeveloperAPI
from ray.rllib.connectors.agent.synced_filter import SyncedFilterAgentConnector
@PublicAPI(stability='alpha')
def create_connectors_for_policy(policy: 'Policy', config: 'AlgorithmConfig'):
    """Util to create agent and action connectors for a Policy.

    Args:
        policy: Policy instance.
        config: Algorithm config dict.
    """
    ctx: ConnectorContext = ConnectorContext.from_policy(policy)
    assert policy.agent_connectors is None and policy.action_connectors is None, 'Can not create connectors for a policy that already has connectors.'
    policy.agent_connectors = get_agent_connectors_from_config(ctx, config)
    policy.action_connectors = get_action_connectors_from_config(ctx, config)
    logger.info('Using connectors:')
    logger.info(policy.agent_connectors.__str__(indentation=4))
    logger.info(policy.action_connectors.__str__(indentation=4))