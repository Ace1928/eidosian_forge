import re
from typing import FrozenSet, NewType, Tuple, Union, cast
from .tags import Tag, parse_tag
from .version import InvalidVersion, Version
def canonicalize_version(version: Union[Version, str], *, strip_trailing_zero: bool=True) -> str:
    """
    This is very similar to Version.__str__, but has one subtle difference
    with the way it handles the release segment.
    """
    if isinstance(version, str):
        try:
            parsed = Version(version)
        except InvalidVersion:
            return version
    else:
        parsed = version
    parts = []
    if parsed.epoch != 0:
        parts.append(f'{parsed.epoch}!')
    release_segment = '.'.join((str(x) for x in parsed.release))
    if strip_trailing_zero:
        release_segment = re.sub('(\\.0)+$', '', release_segment)
    parts.append(release_segment)
    if parsed.pre is not None:
        parts.append(''.join((str(x) for x in parsed.pre)))
    if parsed.post is not None:
        parts.append(f'.post{parsed.post}')
    if parsed.dev is not None:
        parts.append(f'.dev{parsed.dev}')
    if parsed.local is not None:
        parts.append(f'+{parsed.local}')
    return ''.join(parts)