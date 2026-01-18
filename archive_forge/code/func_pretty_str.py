import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def pretty_str(envinfo):

    def replace_nones(dct, replacement='Could not collect'):
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true='Yes', false='No'):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag='[prepend]'):
        lines = text.split('\n')
        updated_lines = [tag + line for line in lines]
        return '\n'.join(updated_lines)

    def replace_if_empty(text, replacement='No relevant packages'):
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string):
        if string is not None and len(string.split('\n')) > 1:
            return '\n{}\n'.format(string)
        return string
    mutable_dict = envinfo._asdict()
    mutable_dict['nvidia_gpu_models'] = maybe_start_on_next_line(envinfo.nvidia_gpu_models)
    dynamic_cuda_fields = ['cuda_runtime_version', 'nvidia_gpu_models', 'nvidia_driver_version']
    all_cuda_fields = dynamic_cuda_fields + ['cudnn_version']
    all_dynamic_cuda_fields_missing = all((mutable_dict[field] is None for field in dynamic_cuda_fields))
    if TORCH_AVAILABLE and (not torch.cuda.is_available()) and all_dynamic_cuda_fields_missing:
        for field in all_cuda_fields:
            mutable_dict[field] = 'No CUDA'
        if envinfo.cuda_compiled_version is None:
            mutable_dict['cuda_compiled_version'] = 'None'
    mutable_dict = replace_bools(mutable_dict)
    mutable_dict = replace_nones(mutable_dict)
    mutable_dict['pip_packages'] = replace_if_empty(mutable_dict['pip_packages'])
    mutable_dict['conda_packages'] = replace_if_empty(mutable_dict['conda_packages'])
    if mutable_dict['pip_packages']:
        mutable_dict['pip_packages'] = prepend(mutable_dict['pip_packages'], '[{}] '.format(envinfo.pip_version))
    if mutable_dict['conda_packages']:
        mutable_dict['conda_packages'] = prepend(mutable_dict['conda_packages'], '[conda] ')
    mutable_dict['cpu_info'] = envinfo.cpu_info
    return env_info_fmt.format(**mutable_dict)