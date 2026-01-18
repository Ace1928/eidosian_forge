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
def get_repo_type(repo_id, repo_type=None, **hub_kwargs):
    if repo_type is not None:
        return repo_type
    try:
        hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type='space', **hub_kwargs)
        return 'space'
    except RepositoryNotFoundError:
        try:
            hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type='model', **hub_kwargs)
            return 'model'
        except RepositoryNotFoundError:
            raise EnvironmentError(f'`{repo_id}` does not seem to be a valid repo identifier on the Hub.')
        except Exception:
            return 'model'
    except Exception:
        return 'space'