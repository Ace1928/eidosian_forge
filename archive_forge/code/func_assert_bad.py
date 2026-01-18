import builtins
import sys
import types
from unittest import SkipTest, mock
import pytest
from packaging.version import Version
from nibabel.optpkg import optional_package
from nibabel.tripwire import TripWire, TripWireError
def assert_bad(pkg_name, min_version=None):
    pkg, have_pkg, setup = optional_package(pkg_name, min_version=min_version)
    assert not have_pkg
    assert isinstance(pkg, TripWire)
    with pytest.raises(TripWireError):
        pkg.a_method
    with pytest.raises(SkipTest):
        setup()