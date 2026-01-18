from .array import ARRAY
from .array import array as _pg_array
from .operators import ASTEXT
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import DELETE_PATH
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from .operators import JSONPATH_ASTEXT
from .operators import PATH_EXISTS
from .operators import PATH_MATCH
from ... import types as sqltypes
from ...sql import cast
def delete_path(self, array):
    """JSONB expression. Deletes field or array element specified in
            the argument array.

            The input may be a list of strings that will be coerced to an
            ``ARRAY`` or an instance of :meth:`_postgres.array`.

            .. versionadded:: 2.0
            """
    if not isinstance(array, _pg_array):
        array = _pg_array(array)
    right_side = cast(array, ARRAY(sqltypes.TEXT))
    return self.operate(DELETE_PATH, right_side, result_type=JSONB)