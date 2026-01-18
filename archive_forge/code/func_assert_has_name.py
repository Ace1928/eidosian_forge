from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Set, Union, Tuple
from triad.collections import IndexedOrderedDict
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.schema import unquote_name
def assert_has_name(self) -> None:
    assert_or_throw(self.has_name, 'column does not have a name')