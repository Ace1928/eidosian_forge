import inspect
import re
import urllib
from typing import List as LList
from typing import Optional, Union
from .... import __version__ as wandb_ver
from .... import termwarn
from ...public import Api as PublicApi
from ._panels import UnknownPanel, WeavePanel, panel_mapping, weave_panels
from .runset import Runset
from .util import (
from .validators import OneOf, TypeValidator
def groupid_to_ordertuple(groupid):
    rs = self.runsets[0]
    if '-run:' in groupid:
        id, rest = groupid.split('-run:', 1)
    else:
        id, rest = groupid.split('-', 1)
    kvs = rest.split('-')
    kvs = [rs.pm_query_generator.pc_back_to_front(v) for v in kvs]
    keys, ordertuple = zip(*[kv.split(':') for kv in kvs])
    rs_name = self._get_rs_by_id(id).name
    return (rs_name, *ordertuple)