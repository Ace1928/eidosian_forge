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
def special_back_to_front(self, name):
    if name is not None:
        kind, rest = name.split(':', 1)
        if kind == 'config':
            pieces = rest.split('.')
            if len(pieces) <= 1:
                raise ValueError(f'Invalid name: {name}')
            elif len(pieces) == 2:
                name = pieces[0]
            elif len(pieces) >= 3:
                name = pieces[:1] + pieces[2:]
                name = '.'.join(name)
            return f'c::{name}'
        elif kind == 'summary':
            name = rest
            return f's::{name}'
    if name is None:
        return name
    elif 'summary:' in name:
        name = name.replace('summary:', '')
        return self.FRONTEND_NAME_MAPPING_REVERSED.get(name, name)
    elif 'run:' in name:
        name = name.replace('run:', '')
        return self.RUN_MAPPING_REVERSED[name]
    return name