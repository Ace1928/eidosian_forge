import copy
import json
import os
import warnings
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import numpy as np
from .dynamic_module_utils import custom_object_save
from .utils import (
@classmethod
def get_feature_extractor_dict(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor object.
        """
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    token = kwargs.pop('token', None)
    use_auth_token = kwargs.pop('use_auth_token', None)
    local_files_only = kwargs.pop('local_files_only', False)
    revision = kwargs.pop('revision', None)
    if use_auth_token is not None:
        warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
        if token is not None:
            raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
        token = use_auth_token
    from_pipeline = kwargs.pop('_from_pipeline', None)
    from_auto_class = kwargs.pop('_from_auto', False)
    user_agent = {'file_type': 'feature extractor', 'from_auto_class': from_auto_class}
    if from_pipeline is not None:
        user_agent['using_pipeline'] = from_pipeline
    if is_offline_mode() and (not local_files_only):
        logger.info('Offline mode: forcing local_files_only=True')
        local_files_only = True
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if os.path.isdir(pretrained_model_name_or_path):
        feature_extractor_file = os.path.join(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME)
    if os.path.isfile(pretrained_model_name_or_path):
        resolved_feature_extractor_file = pretrained_model_name_or_path
        is_local = True
    elif is_remote_url(pretrained_model_name_or_path):
        feature_extractor_file = pretrained_model_name_or_path
        resolved_feature_extractor_file = download_url(pretrained_model_name_or_path)
    else:
        feature_extractor_file = FEATURE_EXTRACTOR_NAME
        try:
            resolved_feature_extractor_file = cached_file(pretrained_model_name_or_path, feature_extractor_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, user_agent=user_agent, revision=revision)
        except EnvironmentError:
            raise
        except Exception:
            raise EnvironmentError(f"Can't load feature extractor for '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a {FEATURE_EXTRACTOR_NAME} file")
    try:
        with open(resolved_feature_extractor_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        feature_extractor_dict = json.loads(text)
    except json.JSONDecodeError:
        raise EnvironmentError(f"It looks like the config file at '{resolved_feature_extractor_file}' is not a valid JSON file.")
    if is_local:
        logger.info(f'loading configuration file {resolved_feature_extractor_file}')
    else:
        logger.info(f'loading configuration file {feature_extractor_file} from cache at {resolved_feature_extractor_file}')
    if 'auto_map' in feature_extractor_dict and (not is_local):
        feature_extractor_dict['auto_map'] = add_model_info_to_auto_map(feature_extractor_dict['auto_map'], pretrained_model_name_or_path)
    return (feature_extractor_dict, kwargs)