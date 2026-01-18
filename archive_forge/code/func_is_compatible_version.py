from typing import Optional
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
def is_compatible_version(version: str, constraint: str, prereleases: bool=True) -> Optional[bool]:
    """Check if a version (e.g. "2.0.0") is compatible given a version
    constraint (e.g. ">=1.9.0,<2.2.1"). If the constraint is a specific version,
    it's interpreted as =={version}.

    version (str): The version to check.
    constraint (str): The constraint string.
    prereleases (bool): Whether to allow prereleases. If set to False,
        prerelease versions will be considered incompatible.
    RETURNS (bool / None): Whether the version is compatible, or None if the
        version or constraint are invalid.
    """
    if constraint[0].isdigit():
        constraint = f'=={constraint}'
    try:
        spec = SpecifierSet(constraint)
        version = Version(version)
    except (InvalidSpecifier, InvalidVersion):
        return None
    spec.prereleases = prereleases
    return version in spec