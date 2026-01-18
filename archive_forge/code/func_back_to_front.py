import ast
import json
import sys
import urllib
from wandb_gql import gql
import wandb
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
def back_to_front(self, name):
    if name in self.FRONTEND_NAME_MAPPING_REVERSED:
        return self.FRONTEND_NAME_MAPPING_REVERSED[name]
    return name