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
def run_name_to_id(name):
    for rs in self.runsets:
        runs = PublicApi().runs(path=f'{rs.entity}/{rs.project}', filters={'display_name': name})
        if len(runs) > 1:
            termwarn('Multiple runs with the same name found! Using the first one.')
        for run in runs:
            if run.name == name:
                return run.id
    raise ValueError('Unable to find this run!')