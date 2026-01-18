import filecmp
import importlib
import os
import re
import shutil
import signal
import sys
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import try_to_load_from_cache
from .utils import (
def custom_object_save(obj: Any, folder: Union[str, os.PathLike], config: Optional[Dict]=None) -> List[str]:
    """
    Save the modeling files corresponding to a custom model/configuration/tokenizer etc. in a given folder. Optionally
    adds the proper fields in a config.

    Args:
        obj (`Any`): The object for which to save the module files.
        folder (`str` or `os.PathLike`): The folder where to save.
        config (`PretrainedConfig` or dictionary, `optional`):
            A config in which to register the auto_map corresponding to this custom object.

    Returns:
        `List[str]`: The list of files saved.
    """
    if obj.__module__ == '__main__':
        logger.warning(f"We can't save the code defining {obj} in {folder} as it's been defined in __main__. You should put this code in a separate module so we can include it in the saved folder and make it easier to share via the Hub.")
        return

    def _set_auto_map_in_config(_config):
        module_name = obj.__class__.__module__
        last_module = module_name.split('.')[-1]
        full_name = f'{last_module}.{obj.__class__.__name__}'
        if 'Tokenizer' in full_name:
            slow_tokenizer_class = None
            fast_tokenizer_class = None
            if obj.__class__.__name__.endswith('Fast'):
                fast_tokenizer_class = f'{last_module}.{obj.__class__.__name__}'
                if getattr(obj, 'slow_tokenizer_class', None) is not None:
                    slow_tokenizer = getattr(obj, 'slow_tokenizer_class')
                    slow_tok_module_name = slow_tokenizer.__module__
                    last_slow_tok_module = slow_tok_module_name.split('.')[-1]
                    slow_tokenizer_class = f'{last_slow_tok_module}.{slow_tokenizer.__name__}'
            else:
                slow_tokenizer_class = f'{last_module}.{obj.__class__.__name__}'
            full_name = (slow_tokenizer_class, fast_tokenizer_class)
        if isinstance(_config, dict):
            auto_map = _config.get('auto_map', {})
            auto_map[obj._auto_class] = full_name
            _config['auto_map'] = auto_map
        elif getattr(_config, 'auto_map', None) is not None:
            _config.auto_map[obj._auto_class] = full_name
        else:
            _config.auto_map = {obj._auto_class: full_name}
    if isinstance(config, (list, tuple)):
        for cfg in config:
            _set_auto_map_in_config(cfg)
    elif config is not None:
        _set_auto_map_in_config(config)
    result = []
    object_file = sys.modules[obj.__module__].__file__
    dest_file = Path(folder) / Path(object_file).name
    shutil.copy(object_file, dest_file)
    result.append(dest_file)
    for needed_file in get_relative_import_files(object_file):
        dest_file = Path(folder) / Path(needed_file).name
        shutil.copy(needed_file, dest_file)
        result.append(dest_file)
    return result