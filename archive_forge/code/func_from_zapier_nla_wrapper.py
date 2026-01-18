from typing import List
from langchain_core._api import warn_deprecated
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.zapier.tool import ZapierNLARunAction
from langchain_community.utilities.zapier import ZapierNLAWrapper
@classmethod
def from_zapier_nla_wrapper(cls, zapier_nla_wrapper: ZapierNLAWrapper) -> 'ZapierToolkit':
    """Create a toolkit from a ZapierNLAWrapper."""
    actions = zapier_nla_wrapper.list()
    tools = [ZapierNLARunAction(action_id=action['id'], zapier_description=action['description'], params_schema=action['params'], api_wrapper=zapier_nla_wrapper) for action in actions]
    return cls(tools=tools)