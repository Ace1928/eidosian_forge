from __future__ import absolute_import, division, print_function
import sys
import math
from collections import OrderedDict
from datetime import datetime, date, time
from decimal import Decimal
from petl.compat import izip_longest, text_type, string_types, PY3
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.transform.headers import skip, setheader
from petl.util.base import Table, dicts, fieldnames, iterpeek, wrap
def fromavro(source, limit=None, skips=0, **avro_args):
    """Extract a table from the records of a avro file.

    The `source` argument (string or file-like or fastavro.reader) can either
    be  the path of the file, a file-like input stream or a instance from
    fastavro.reader.

    The `limit` and `skip` arguments can be used to limit the range of rows 
    to extract.

    The `sample` argument (int, optional) defines how many rows are inspected
    for discovering the field types and building a schema for the avro file 
    when the `schema` argument is not passed.

    The rows fields read from file can have scalar values like int, string,
    float, datetime, date and decimal but can also have compound types like 
    enum, :ref:`array <array_schema>`, map, union and record. 
    The fields types can also have recursive structures defined 
    in :ref:`complex schemas <complex_schema>`.

    Also types with :ref:`logical types <logical_schema>` types are read and 
    translated to coresponding python types: long timestamp-millis and 
    long timestamp-micros: datetime.datetime, int date: datetime.date, 
    bytes decimal and fixed decimal: Decimal, int time-millis and 
    long time-micros: datetime.time.

    Example usage for reading files::

        >>> # set up a Avro file to demonstrate with
        ...
        >>> schema1 = {
        ...     'doc': 'Some people records.',
        ...     'name': 'People',
        ...     'namespace': 'test',
        ...     'type': 'record',
        ...     'fields': [
        ...         {'name': 'name', 'type': 'string'},
        ...         {'name': 'friends', 'type': 'int'},
        ...         {'name': 'age', 'type': 'int'},
        ...     ]
        ... }
        ...
        >>> records1 = [
        ...     {'name': 'Bob', 'friends': 42, 'age': 33},
        ...     {'name': 'Jim', 'friends': 13, 'age': 69},
        ...     {'name': 'Joe', 'friends': 86, 'age': 17},
        ...     {'name': 'Ted', 'friends': 23, 'age': 51}
        ... ]
        ...
        >>> import fastavro
        >>> parsed_schema1 = fastavro.parse_schema(schema1)
        >>> with open('example.file1.avro', 'wb') as f1:
        ...     fastavro.writer(f1, parsed_schema1, records1)
        ...
        >>> # now demonstrate the use of fromavro()
        >>> import petl as etl
        >>> tbl1 = etl.fromavro('example.file1.avro')
        >>> tbl1
        +-------+---------+-----+
        | name  | friends | age |
        +=======+=========+=====+
        | 'Bob' |      42 |  33 |
        +-------+---------+-----+
        | 'Jim' |      13 |  69 |
        +-------+---------+-----+
        | 'Joe' |      86 |  17 |
        +-------+---------+-----+
        | 'Ted' |      23 |  51 |
        +-------+---------+-----+

    .. versionadded:: 1.4.0

    """
    source2 = read_source_from_arg(source)
    return AvroView(source=source2, limit=limit, skips=skips, **avro_args)