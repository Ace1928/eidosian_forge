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
@classmethod
def from_report_urls(cls, urls: LList[str]) -> 'Gallery':
    from .report import Report
    ids = [Report._url_to_report_id(url) for url in urls]
    return cls(ids)