import importlib
import inspect
import itertools
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import huggingface_hub
from packaging import version
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import SAFE_WEIGHTS_NAME, TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from ..utils import CONFIG_NAME
from ..utils.import_utils import is_onnx_available
@classmethod
def standardize_model_attributes(cls, model: Union['PreTrainedModel', 'TFPreTrainedModel'], library_name: Optional[str]=None):
    """
        Updates the model for export. This function is suitable to make required changes to the models from different
        libraries to follow transformers style.

        Args:
            model_name_or_path (`Union[str, Path]`):
                Can be either the model id of a model repo on the Hugging Face Hub, or a path to a local directory
                containing a model.
            model (`Union[PreTrainedModel, TFPreTrainedModel]`):
                The instance of the model.
            subfolder (`str`, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`, *optional*, defaults to `None`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Optional[str]`, *optional*):
                Path to a directory in which a downloaded pretrained model weights have been cached if the standard cache should not be used.
            library_name (`Optional[str]`, *optional*)::
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers".
        """
    library_name = TasksManager._infer_library_from_model(model, library_name)
    if library_name == 'diffusers':
        model.config.export_model_type = 'stable-diffusion'
    elif library_name == 'timm':
        model_config = PretrainedConfig.from_dict(model.pretrained_cfg)
        setattr(model, 'config', model_config)
        model.config.export_model_type = model.pretrained_cfg['architecture']
    elif library_name == 'sentence_transformers':
        if 'Transformer' in model[0].__class__.__name__:
            model.config = model[0].auto_model.config
            model.config.export_model_type = 'transformer'
        elif 'CLIP' in model[0].__class__.__name__:
            model.config = model[0].model.config
            model.config.export_model_type = 'clip'
        else:
            raise ValueError(f'The export of a sentence_transformers model with the first module being {model[0].__class__.__name__} is currently not supported in Optimum. Please open an issue or submit a PR to add the support.')