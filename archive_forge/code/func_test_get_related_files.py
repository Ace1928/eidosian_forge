import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_get_related_files(_temp_analyze_files):
    orig_img, orig_hdr = _temp_analyze_files
    related_files = get_related_files(orig_img)
    assert orig_img in related_files
    assert orig_hdr in related_files
    related_files = get_related_files(orig_hdr)
    assert orig_img in related_files
    assert orig_hdr in related_files