from typing import Any, Generic, Iterable, Iterator, List, Optional, Set, Tuple, TypeVar
@property
def drop_duplicate_entries(self) -> bool:
    return self._drop_set is not None