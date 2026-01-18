import logging
from collections import OrderedDict
from typing import Dict, List
from pip._vendor.packaging.specifiers import LegacySpecifier
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import LegacyVersion
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.deprecation import deprecated
def has_requirement(self, name: str) -> bool:
    project_name = canonicalize_name(name)
    return project_name in self.requirements and (not self.requirements[project_name].constraint)