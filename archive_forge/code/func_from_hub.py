import base64
import importlib
import inspect
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import create_repo, hf_hub_download, metadata_update, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, get_session
from ..dynamic_module_utils import custom_object_save, get_class_from_dynamic_module, get_imports
from ..image_utils import is_pil_image
from ..models.auto import AutoProcessor
from ..utils import (
from .agent_types import handle_agent_inputs, handle_agent_outputs
from {module_name} import {class_name}
@classmethod
def from_hub(cls, repo_id: str, model_repo_id: Optional[str]=None, token: Optional[str]=None, remote: bool=False, **kwargs):
    """
        Loads a tool defined on the Hub.

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            model_repo_id (`str`, *optional*):
                If your tool uses a model and you want to use a different model than the default, you can pass a second
                repo ID or an endpoint url to this argument.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            remote (`bool`, *optional*, defaults to `False`):
                Whether to use your tool by downloading the model or (if it is available) with an inference endpoint.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the
                others will be passed along to its init.
        """
    if remote and model_repo_id is None:
        endpoints = get_default_endpoints()
        if repo_id not in endpoints:
            raise ValueError(f'Could not infer a default endpoint for {repo_id}, you need to pass one using the `model_repo_id` argument.')
        model_repo_id = endpoints[repo_id]
    hub_kwargs_names = ['cache_dir', 'force_download', 'resume_download', 'proxies', 'revision', 'repo_type', 'subfolder', 'local_files_only']
    hub_kwargs = {k: v for k, v in kwargs.items() if k in hub_kwargs_names}
    hub_kwargs['repo_type'] = get_repo_type(repo_id, **hub_kwargs)
    resolved_config_file = cached_file(repo_id, TOOL_CONFIG_FILE, token=token, **hub_kwargs, _raise_exceptions_for_gated_repo=False, _raise_exceptions_for_missing_entries=False, _raise_exceptions_for_connection_errors=False)
    is_tool_config = resolved_config_file is not None
    if resolved_config_file is None:
        resolved_config_file = cached_file(repo_id, CONFIG_NAME, token=token, **hub_kwargs, _raise_exceptions_for_gated_repo=False, _raise_exceptions_for_missing_entries=False, _raise_exceptions_for_connection_errors=False)
    if resolved_config_file is None:
        raise EnvironmentError(f'{repo_id} does not appear to provide a valid configuration in `tool_config.json` or `config.json`.')
    with open(resolved_config_file, encoding='utf-8') as reader:
        config = json.load(reader)
    if not is_tool_config:
        if 'custom_tool' not in config:
            raise EnvironmentError(f'{repo_id} does not provide a mapping to custom tools in its configuration `config.json`.')
        custom_tool = config['custom_tool']
    else:
        custom_tool = config
    tool_class = custom_tool['tool_class']
    tool_class = get_class_from_dynamic_module(tool_class, repo_id, token=token, **hub_kwargs)
    if len(tool_class.name) == 0:
        tool_class.name = custom_tool['name']
    if tool_class.name != custom_tool['name']:
        logger.warning(f'{tool_class.__name__} implements a different name in its configuration and class. Using the tool configuration name.')
        tool_class.name = custom_tool['name']
    if len(tool_class.description) == 0:
        tool_class.description = custom_tool['description']
    if tool_class.description != custom_tool['description']:
        logger.warning(f'{tool_class.__name__} implements a different description in its configuration and class. Using the tool configuration description.')
        tool_class.description = custom_tool['description']
    if remote:
        return RemoteTool(model_repo_id, token=token, tool_class=tool_class)
    return tool_class(model_repo_id, token=token, **kwargs)