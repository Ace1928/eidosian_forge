import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
def download_h5(run_id, entity=None, project=None, out_dir=None):
    api = Api()
    meta = api.download_url(project or api.settings('project'), DEEP_SUMMARY_FNAME, entity=entity or api.settings('entity'), run=run_id)
    if meta and 'md5' in meta and (meta['md5'] is not None):
        wandb.termlog('Downloading summary data...')
        path, res = api.download_write_file(meta, out_dir=out_dir)
        return path