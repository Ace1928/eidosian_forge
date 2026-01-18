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
def launch_gradio_demo(tool_class: Tool):
    """
    Launches a gradio demo for a tool. The corresponding tool class needs to properly implement the class attributes
    `inputs` and `outputs`.

    Args:
        tool_class (`type`): The class of the tool for which to launch the demo.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError('Gradio should be installed in order to launch a gradio demo.')
    tool = tool_class()

    def fn(*args, **kwargs):
        return tool(*args, **kwargs)
    gr.Interface(fn=fn, inputs=tool_class.inputs, outputs=tool_class.outputs, title=tool_class.__name__, article=tool.description).launch()