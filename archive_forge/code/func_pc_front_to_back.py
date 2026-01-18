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
def pc_front_to_back(self, name):
    name, *rest = name.split('.')
    rest = '.' + '.'.join(rest) if rest else ''
    if name is None:
        return None
    elif name in self.panel_metrics_helper.FRONTEND_NAME_MAPPING:
        return 'summary:' + self.panel_metrics_helper.FRONTEND_NAME_MAPPING[name]
    elif name in self.FRONTEND_NAME_MAPPING:
        return self.FRONTEND_NAME_MAPPING[name]
    elif name in self.FRONTEND_NAME_MAPPING_REVERSED:
        return name
    elif name in self.run_set._runs_config:
        return f'config:{name}.value{rest}'
    else:
        return f'summary:{name}{rest}'