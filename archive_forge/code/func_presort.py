import json
from typing import Any, Dict, List, Tuple
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_size
from triad.utils.hash import to_uuid
from triad.utils.pyarrow import SchemaedDataPartitioner
from triad.utils.schema import safe_split_out_of_quote, unquote_name
@property
def presort(self) -> IndexedOrderedDict[str, bool]:
    """Get presort pairs of the spec

        .. admonition:: Examples

            >>> p = PartitionSpec(by=["a"],presort="b,c desc")
            >>> assert p.presort == {"b":True, "c":False}
        """
    return self._presort