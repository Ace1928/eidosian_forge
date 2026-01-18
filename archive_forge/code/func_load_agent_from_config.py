import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union
import yaml
from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool
from langchain.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.types import AGENT_TO_CLASS
from langchain.chains.loading import load_chain, load_chain_from_config
@deprecated('0.1.0', removal='0.2.0')
def load_agent_from_config(config: dict, llm: Optional[BaseLanguageModel]=None, tools: Optional[List[Tool]]=None, **kwargs: Any) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]:
    """Load agent from Config Dict.

    Args:
        config: Config dict to load agent from.
        llm: Language model to use as the agent.
        tools: List of tools this agent has access to.
        **kwargs: Additional keyword arguments passed to the agent executor.

    Returns:
        An agent executor.
    """
    if '_type' not in config:
        raise ValueError('Must specify an agent Type in config')
    load_from_tools = config.pop('load_from_llm_and_tools', False)
    if load_from_tools:
        if llm is None:
            raise ValueError('If `load_from_llm_and_tools` is set to True, then LLM must be provided')
        if tools is None:
            raise ValueError('If `load_from_llm_and_tools` is set to True, then tools must be provided')
        return _load_agent_from_tools(config, llm, tools, **kwargs)
    config_type = config.pop('_type')
    if config_type not in AGENT_TO_CLASS:
        raise ValueError(f'Loading {config_type} agent not supported')
    agent_cls = AGENT_TO_CLASS[config_type]
    if 'llm_chain' in config:
        config['llm_chain'] = load_chain_from_config(config.pop('llm_chain'))
    elif 'llm_chain_path' in config:
        config['llm_chain'] = load_chain(config.pop('llm_chain_path'))
    else:
        raise ValueError('One of `llm_chain` and `llm_chain_path` should be specified.')
    if 'output_parser' in config:
        logger.warning('Currently loading output parsers on agent is not supported, will just use the default one.')
        del config['output_parser']
    combined_config = {**config, **kwargs}
    return agent_cls(**combined_config)