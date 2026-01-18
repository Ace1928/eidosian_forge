from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
@property
def as_name(self) -> str:
    """The name assigned by :meth:`~.alias`

        :return: the alias

        .. admonition:: Examples

            .. code-block:: python

                assert "" == col("a").as_name
                assert "b" == col("a").alias("b").as_name
                assert "x" == (col("a") * 2).alias("x").as_name
        """
    return self._as_name