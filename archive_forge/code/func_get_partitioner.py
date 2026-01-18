import json
from typing import Any, Dict, List, Tuple
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_size
from triad.utils.hash import to_uuid
from triad.utils.pyarrow import SchemaedDataPartitioner
from triad.utils.schema import safe_split_out_of_quote, unquote_name
def get_partitioner(self, schema: Schema) -> SchemaedDataPartitioner:
    """Get :class:`~triad.utils.pyarrow.SchemaedDataPartitioner` by input
        dataframe schema

        :param schema: the dataframe schema this partition spec to operate on
        :return: SchemaedDataPartitioner object
        """
    pos = [schema.index_of_key(key) for key in self.partition_by]
    return SchemaedDataPartitioner(schema.pa_schema, pos, sizer=None, row_limit=self._row_limit, size_limit=self._size_limit)