from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
def test_definition_kwonlyargs():
    i = inspector.info(f_kwarg, oname='f_kwarg')
    assert i['definition'] == 'f_kwarg(pos, *, kwonly)'