import json
from typing import Any, Dict, List, Tuple
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_size
from triad.utils.hash import to_uuid
from triad.utils.pyarrow import SchemaedDataPartitioner
from triad.utils.schema import safe_split_out_of_quote, unquote_name
def get_sorts(self, schema: Schema, with_partition_keys: bool=True) -> IndexedOrderedDict[str, bool]:
    """Get keys for sorting in a partition, it's the combination of partition
        keys plus the presort keys

        :param schema: the dataframe schema this partition spec to operate on
        :param with_partition_keys: whether to include partition keys
        :return: an ordered dictionary of key, order pairs

        .. admonition:: Examples

            >>> p = PartitionSpec(by=["a"],presort="b , c dESc")
            >>> schema = Schema("a:int,b:int,c:int,d:int"))
            >>> assert p.get_sorts(schema) == {"a":True, "b":True, "c": False}
        """
    d: IndexedOrderedDict[str, bool] = IndexedOrderedDict()
    if with_partition_keys:
        for p in self.partition_by:
            aot(p in schema, lambda: KeyError(f'{p} not in {schema}'))
            d[p] = True
    for p, v in self.presort.items():
        aot(p in schema, lambda: KeyError(f'{p} not in {schema}'))
        d[p] = v
    return d