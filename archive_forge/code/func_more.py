import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
@property
def more(self):
    if self.last_response:
        return self.last_response['project']['artifactType']['artifact']['files']['pageInfo']['hasNextPage']
    else:
        return True