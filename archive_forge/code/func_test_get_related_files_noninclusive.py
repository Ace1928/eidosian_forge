import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_get_related_files_noninclusive(_temp_analyze_files):
    orig_img, orig_hdr = _temp_analyze_files
    related_files = get_related_files(orig_img, include_this_file=False)
    assert orig_img not in related_files
    assert orig_hdr in related_files
    related_files = get_related_files(orig_hdr, include_this_file=False)
    assert orig_img in related_files
    assert orig_hdr not in related_files