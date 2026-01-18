from typing import FrozenSet, Iterable, Optional, Tuple, Union
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName
from pip._vendor.packaging.version import LegacyVersion, Version
from pip._internal.models.link import Link, links_equivalent
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.hashes import Hashes
@classmethod
def from_ireq(cls, ireq: InstallRequirement) -> 'Constraint':
    links = frozenset([ireq.link]) if ireq.link else frozenset()
    return Constraint(ireq.specifier, ireq.hashes(trust_internet=False), links)