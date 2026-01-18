import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
def artifact_collection_edge_name(server_supports_artifact_collections: bool) -> str:
    return 'artifactCollection' if server_supports_artifact_collections else 'artifactSequence'