import json
from typing import Any, Dict, List, Tuple
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_size
from triad.utils.hash import to_uuid
from triad.utils.pyarrow import SchemaedDataPartitioner
from triad.utils.schema import safe_split_out_of_quote, unquote_name
def get_cursor(self, schema: Schema, physical_partition_no: int) -> 'PartitionCursor':
    """Get :class:`.PartitionCursor` based on
        dataframe schema and physical partition number. You normally don't call
        this method directly

        :param schema: the dataframe schema this partition spec to operate on
        :param physical_partition_no: physical partition no passed in by
          :class:`~fugue.execution.execution_engine.ExecutionEngine`
        :return: PartitionCursor object
        """
    return PartitionCursor(schema, self, physical_partition_no)