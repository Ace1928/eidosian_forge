import importlib
import logging
import os
import sys
import tempfile
from typing import Optional
import torch
from torch.distributed.nn.jit.templates.remote_module_template import (
def instantiate_non_scriptable_remote_module_template():
    generated_module_name = f'{_FILE_PREFIX}non_scriptable'
    str_dict = dict(assign_module_interface_cls='module_interface_cls = None', args='*args', kwargs='**kwargs', arg_types='*args, **kwargs', arrow_and_return_type='', arrow_and_future_return_type='', jit_script_decorator='')
    return _do_instantiate_remote_module_template(generated_module_name, str_dict, True)