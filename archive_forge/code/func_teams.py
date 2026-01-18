import requests
from wandb_gql import gql
import wandb
from wandb.apis.attrs import Attrs
@property
def teams(self):
    if self._attrs.get('teams') is None:
        return []
    return [k['node']['name'] for k in self._attrs['teams']['edges']]