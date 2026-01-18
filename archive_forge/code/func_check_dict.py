import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def check_dict(ref_dict, tst_dict):
    """Compare dictionaries of inputs and and those loaded from json files"""

    def to_list(x):
        if isinstance(x, tuple):
            x = list(x)
        if isinstance(x, list):
            for i, xel in enumerate(x):
                x[i] = to_list(xel)
        return x
    failed_dict = {}
    for key, value in list(ref_dict.items()):
        newval = to_list(tst_dict[key])
        if newval != value:
            failed_dict[key] = (value, newval)
    return failed_dict