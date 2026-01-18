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
@DeveloperAPI
def maybe_get_filters_for_syncing(rollout_worker, policy_id):
    policy = rollout_worker.policy_map[policy_id]
    if not policy.agent_connectors:
        return
    filter_connectors = policy.agent_connectors[SyncedFilterAgentConnector]
    if not filter_connectors:
        return
    assert len(filter_connectors) == 1, 'ConnectorPipeline has multiple connectors of type SyncedFilterAgentConnector but can only have one.'
    rollout_worker.filters[policy_id] = filter_connectors[0].filter