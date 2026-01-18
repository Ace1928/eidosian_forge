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
def resolve_trust_remote_code(trust_remote_code, model_name, has_local_code, has_remote_code):
    if trust_remote_code is None:
        if has_local_code:
            trust_remote_code = False
        elif has_remote_code and TIME_OUT_REMOTE_CODE > 0:
            try:
                signal.signal(signal.SIGALRM, _raise_timeout_error)
                signal.alarm(TIME_OUT_REMOTE_CODE)
                while trust_remote_code is None:
                    answer = input(f'The repository for {model_name} contains custom code which must be executed to correctly load the model. You can inspect the repository content at https://hf.co/{model_name}.\nYou can avoid this prompt in future by passing the argument `trust_remote_code=True`.\n\nDo you wish to run the custom code? [y/N] ')
                    if answer.lower() in ['yes', 'y', '1']:
                        trust_remote_code = True
                    elif answer.lower() in ['no', 'n', '0', '']:
                        trust_remote_code = False
                signal.alarm(0)
            except Exception:
                raise ValueError(f'The repository for {model_name} contains custom code which must be executed to correctly load the model. You can inspect the repository content at https://hf.co/{model_name}.\nPlease pass the argument `trust_remote_code=True` to allow custom code to be run.')
        elif has_remote_code:
            _raise_timeout_error(None, None)
    if has_remote_code and (not has_local_code) and (not trust_remote_code):
        raise ValueError(f'Loading {model_name} requires you to execute the configuration file in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set the option `trust_remote_code=True` to remove this error.')
    return trust_remote_code