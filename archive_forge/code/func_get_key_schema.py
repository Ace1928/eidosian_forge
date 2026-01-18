import json
from typing import Any, Dict, List, Tuple
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_size
from triad.utils.hash import to_uuid
from triad.utils.pyarrow import SchemaedDataPartitioner
from triad.utils.schema import safe_split_out_of_quote, unquote_name
def get_key_schema(self, schema: Schema) -> Schema:
    """Get partition keys schema

        :param schema: the dataframe schema this partition spec to operate on
        :return: the sub-schema only containing partition keys
        """
    return schema.extract(self.partition_by)