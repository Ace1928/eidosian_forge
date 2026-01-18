import builtins
import sys
import types
from unittest import SkipTest, mock
import pytest
from packaging.version import Version
from nibabel.optpkg import optional_package
from nibabel.tripwire import TripWire, TripWireError
def assert_good(pkg_name, min_version=None):
    pkg, have_pkg, setup = optional_package(pkg_name, min_version=min_version)
    assert have_pkg
    assert sys.modules[pkg_name] == pkg
    assert setup() is None