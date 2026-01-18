import logging
from collections import OrderedDict
from typing import Dict, List
from pip._vendor.packaging.specifiers import LegacySpecifier
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import LegacyVersion
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.deprecation import deprecated
def add_unnamed_requirement(self, install_req: InstallRequirement) -> None:
    assert not install_req.name
    self.unnamed_requirements.append(install_req)