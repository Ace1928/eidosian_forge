from __future__ import annotations
import json
import pickle  # use pickle, not cPickle so that we get the traceback in case of errors
import string
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from unittest import TestCase
import pytest
from monty.json import MontyDecoder, MontyEncoder, MSONable
from monty.serialization import loadfn
from pymatgen.core import ROOT, SETTINGS, Structure
def assert_msonable(self, obj: MSONable, test_is_subclass: bool=True) -> str:
    """Test if obj is MSONable and verify the contract is fulfilled.

        By default, the method tests whether obj is an instance of MSONable.
        This check can be deactivated by setting test_is_subclass=False.
        """
    if test_is_subclass:
        assert isinstance(obj, MSONable)
    assert obj.as_dict() == type(obj).from_dict(obj.as_dict()).as_dict()
    json_str = json.dumps(obj.as_dict(), cls=MontyEncoder)
    round_trip = json.loads(json_str, cls=MontyDecoder)
    assert issubclass(type(round_trip), type(obj)), f'{type(round_trip)} != {type(obj)}'
    return json_str