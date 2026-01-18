import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
def open_h5(self):
    if not self._h5 and h5py:
        download_h5(self._run.id, entity=self._run.entity, project=self._run.project, out_dir=self._run.dir)
    super().open_h5()