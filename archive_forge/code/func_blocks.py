import base64
import inspect
import json
import re
import urllib
from copy import deepcopy
from typing import List as LList
from .... import __version__ as wandb_ver
from .... import termlog, termwarn
from ....sdk.lib import ipython
from ...public import Api as PublicApi
from ...public import RetryingClient
from ._blocks import P, PanelGrid, UnknownBlock, WeaveBlock, block_mapping, weave_blocks
from .mutations import UPSERT_VIEW, VIEW_REPORT
from .runset import Runset
from .util import Attr, Base, Block, coalesce, generate_name, nested_get, nested_set
from .validators import OneOf, TypeValidator
@blocks.setter
def blocks(self, new_blocks):
    json_path = self._get_path('blocks')
    new_block_specs = [P('').spec] + [b.spec for b in new_blocks] + [P('').spec]
    nested_set(self, json_path, new_block_specs)