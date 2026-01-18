import json
from unittest.mock import Mock, patch
import pytest
from traitlets.config import Config, Configurable
from jupyterlab.extensions import PyPIExtensionManager, ReadOnlyExtensionManager
from jupyterlab.extensions.manager import ExtensionManager, ExtensionPackage, PluginManager
from . import fake_client_factory
@pytest.mark.parametrize('version, expected', (('1', '1'), ('1.0', '1.0'), ('1.0.0', '1.0.0'), ('1.0.0a52', '1.0.0-alpha.52'), ('1.0.0b3', '1.0.0-beta.3'), ('1.0.0rc22', '1.0.0-rc.22'), ('1.0.0rc23.post2', '1.0.0-rc.23'), ('1.0.0rc24.dev2', '1.0.0-rc.24'), ('1.0.0rc25.post4.dev2', '1.0.0-rc.25')))
def test_ExtensionManager_get_semver_version(version, expected):
    assert ExtensionManager.get_semver_version(version) == expected