import os
from .. import get_info
from ..info import get_nipype_gitversion
import pytest
def test_nipype_info():
    exception_not_raised = True
    try:
        get_info()
    except Exception:
        exception_not_raised = False
    assert exception_not_raised