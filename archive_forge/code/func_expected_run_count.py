import urllib
from typing import Optional
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.sdk.lib import ipython
@property
def expected_run_count(self) -> Optional[int]:
    """Return the number of expected runs in the sweep or None for infinite runs."""
    return self._attrs.get('runCountExpected')