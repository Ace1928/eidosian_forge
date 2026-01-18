from typing import FrozenSet, Optional, Set
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import CommandError
@staticmethod
def handle_mutual_excludes(value: str, target: Set[str], other: Set[str]) -> None:
    if value.startswith('-'):
        raise CommandError('--no-binary / --only-binary option requires 1 argument.')
    new = value.split(',')
    while ':all:' in new:
        other.clear()
        target.clear()
        target.add(':all:')
        del new[:new.index(':all:') + 1]
        if ':none:' not in new:
            return
    for name in new:
        if name == ':none:':
            target.clear()
            continue
        name = canonicalize_name(name)
        other.discard(name)
        target.add(name)