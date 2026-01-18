from typing import FrozenSet, Optional, Set
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import CommandError
def get_allowed_formats(self, canonical_name: str) -> FrozenSet[str]:
    result = {'binary', 'source'}
    if canonical_name in self.only_binary:
        result.discard('source')
    elif canonical_name in self.no_binary:
        result.discard('binary')
    elif ':all:' in self.only_binary:
        result.discard('source')
    elif ':all:' in self.no_binary:
        result.discard('binary')
    return frozenset(result)