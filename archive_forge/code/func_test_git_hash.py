import os
from .. import get_info
from ..info import get_nipype_gitversion
import pytest
@pytest.mark.skipif(not get_nipype_gitversion(), reason='not able to get version from get_nipype_gitversion')
def test_git_hash():
    get_nipype_gitversion()[1:] == get_info()['commit_hash']