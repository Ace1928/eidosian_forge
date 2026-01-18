import copy
import json
import os
import warnings
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .image_transforms import center_crop, normalize, rescale
from .image_utils import ChannelDimension
from .utils import (
@classmethod
def get_image_processor_dict(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        image processor of type [`~image_processor_utils.ImageProcessingMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the image processor object.
        """
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    token = kwargs.pop('token', None)
    use_auth_token = kwargs.pop('use_auth_token', None)
    local_files_only = kwargs.pop('local_files_only', False)
    revision = kwargs.pop('revision', None)
    subfolder = kwargs.pop('subfolder', '')
    from_pipeline = kwargs.pop('_from_pipeline', None)
    from_auto_class = kwargs.pop('_from_auto', False)
    if use_auth_token is not None:
        warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
        if token is not None:
            raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
        token = use_auth_token
    user_agent = {'file_type': 'image processor', 'from_auto_class': from_auto_class}
    if from_pipeline is not None:
        user_agent['using_pipeline'] = from_pipeline
    if is_offline_mode() and (not local_files_only):
        logger.info('Offline mode: forcing local_files_only=True')
        local_files_only = True
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if os.path.isdir(pretrained_model_name_or_path):
        image_processor_file = os.path.join(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME)
    if os.path.isfile(pretrained_model_name_or_path):
        resolved_image_processor_file = pretrained_model_name_or_path
        is_local = True
    elif is_remote_url(pretrained_model_name_or_path):
        image_processor_file = pretrained_model_name_or_path
        resolved_image_processor_file = download_url(pretrained_model_name_or_path)
    else:
        image_processor_file = IMAGE_PROCESSOR_NAME
        try:
            resolved_image_processor_file = cached_file(pretrained_model_name_or_path, image_processor_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, user_agent=user_agent, revision=revision, subfolder=subfolder)
        except EnvironmentError:
            raise
        except Exception:
            raise EnvironmentError(f"Can't load image processor for '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a {IMAGE_PROCESSOR_NAME} file")
    try:
        with open(resolved_image_processor_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        image_processor_dict = json.loads(text)
    except json.JSONDecodeError:
        raise EnvironmentError(f"It looks like the config file at '{resolved_image_processor_file}' is not a valid JSON file.")
    if is_local:
        logger.info(f'loading configuration file {resolved_image_processor_file}')
    else:
        logger.info(f'loading configuration file {image_processor_file} from cache at {resolved_image_processor_file}')
    if 'auto_map' in image_processor_dict and (not is_local):
        image_processor_dict['auto_map'] = add_model_info_to_auto_map(image_processor_dict['auto_map'], pretrained_model_name_or_path)
    return (image_processor_dict, kwargs)