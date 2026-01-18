import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
def get_forge(branch, possible_forges=None):
    """Find the forge for a branch.

    :param branch: Branch to find forge for
    :param possible_forges: Optional list of forges to reuse
    :raise UnsupportedForge: if there is no forge that supports `branch`
    :return: A `Forge` object
    """
    if possible_forges:
        for forge in possible_forges:
            if forge.hosts(branch):
                return forge
    for name, forge_cls in forges.items():
        try:
            forge = forge_cls.probe_from_branch(branch)
        except UnsupportedForge:
            pass
        else:
            if possible_forges is not None:
                possible_forges.append(forge)
            return forge
    raise UnsupportedForge(branch)