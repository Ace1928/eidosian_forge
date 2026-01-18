import logging
from collections import OrderedDict
from typing import Dict, List
from pip._vendor.packaging.specifiers import LegacySpecifier
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import LegacyVersion
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.deprecation import deprecated
def add_named_requirement(self, install_req: InstallRequirement) -> None:
    assert install_req.name
    project_name = canonicalize_name(install_req.name)
    self.requirements[project_name] = install_req