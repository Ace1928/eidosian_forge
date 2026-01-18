import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
def convert_objects(self):
    return [public.File(self.client, r['node']) for r in self.last_response['project']['artifactType']['artifact']['files']['edges']]