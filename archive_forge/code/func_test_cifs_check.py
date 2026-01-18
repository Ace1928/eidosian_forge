import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_cifs_check():
    assert isinstance(_cifs_table, list)
    assert isinstance(on_cifs('/'), bool)
    fake_table = [('/scratch/tmp', 'ext4'), ('/scratch', 'cifs')]
    cifs_targets = [('/scratch/tmp/x/y', False), ('/scratch/tmp/x', False), ('/scratch/x/y', True), ('/scratch/x', True), ('/x/y', False), ('/x', False), ('/', False)]
    orig_table = _cifs_table[:]
    _cifs_table[:] = []
    for target, _ in cifs_targets:
        assert on_cifs(target) is False
    _cifs_table.extend(fake_table)
    for target, expected in cifs_targets:
        assert on_cifs(target) is expected
    _cifs_table[:] = []
    _cifs_table.extend(orig_table)