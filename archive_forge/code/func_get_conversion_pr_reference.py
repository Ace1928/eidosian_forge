import json
import uuid
from typing import Optional
import requests
from huggingface_hub import Discussion, HfApi, get_repo_discussions
from .utils import cached_file, logging
def get_conversion_pr_reference(api: HfApi, model_id: str, **kwargs):
    private = api.model_info(model_id).private
    logger.info('Attempting to create safetensors variant')
    pr_title = 'Adding `safetensors` variant of this model'
    token = kwargs.get('token')
    pr = previous_pr(api, model_id, pr_title, token=token)
    if pr is None or (not private and pr.author != 'SFConvertBot'):
        spawn_conversion(token, private, model_id)
        pr = previous_pr(api, model_id, pr_title, token=token)
    else:
        logger.info('Safetensors PR exists')
    sha = f'refs/pr/{pr.num}'
    return sha