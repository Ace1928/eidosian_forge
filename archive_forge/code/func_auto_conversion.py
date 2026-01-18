import json
import uuid
from typing import Optional
import requests
from huggingface_hub import Discussion, HfApi, get_repo_discussions
from .utils import cached_file, logging
def auto_conversion(pretrained_model_name_or_path: str, **cached_file_kwargs):
    api = HfApi(token=cached_file_kwargs.get('token'))
    sha = get_conversion_pr_reference(api, pretrained_model_name_or_path, **cached_file_kwargs)
    if sha is None:
        return (None, None)
    cached_file_kwargs['revision'] = sha
    del cached_file_kwargs['_commit_hash']
    sharded = api.file_exists(pretrained_model_name_or_path, 'model.safetensors.index.json', revision=sha, token=cached_file_kwargs.get('token'))
    filename = 'model.safetensors.index.json' if sharded else 'model.safetensors'
    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
    return (resolved_archive_file, sha, sharded)