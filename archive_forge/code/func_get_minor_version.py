from typing import Optional
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
def get_minor_version(version: str) -> Optional[str]:
    """Get the major + minor version (without patch or prerelease identifiers).

    version (str): The version.
    RETURNS (str): The major + minor version or None if version is invalid.
    """
    try:
        v = Version(version)
    except (TypeError, InvalidVersion):
        return None
    return f'{v.major}.{v.minor}'