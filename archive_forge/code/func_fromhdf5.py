from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from petl.compat import string_types
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, data
from petl.io.numpy import infer_dtype
def fromhdf5(source, where=None, name=None, condition=None, condvars=None, start=None, stop=None, step=None):
    """
    Provides access to an HDF5 table. E.g.::

        >>> import petl as etl
        >>>
        >>> # set up a new hdf5 table to demonstrate with
        >>> class FooBar(tables.IsDescription): # doctest: +SKIP
        ...     foo = tables.Int32Col(pos=0) # doctest: +SKIP
        ...     bar = tables.StringCol(6, pos=2) # doctest: +SKIP
        >>> #
        >>> def setup_hdf5_table():
        ...     import tables
        ...     h5file = tables.open_file('example.h5', mode='w',
        ...                               title='Example file')
        ...     h5file.create_group('/', 'testgroup', 'Test Group')
        ...     h5table = h5file.create_table('/testgroup', 'testtable', FooBar,
        ...                                   'Test Table')
        ...     # load some data into the table
        ...     table1 = (('foo', 'bar'),
        ...               (1, b'asdfgh'),
        ...               (2, b'qwerty'),
        ...               (3, b'zxcvbn'))
        ...     for row in table1[1:]:
        ...         for i, f in enumerate(table1[0]):
        ...             h5table.row[f] = row[i]
        ...         h5table.row.append()
        ...     h5file.flush()
        ...     h5file.close()
        >>>
        >>> setup_hdf5_table() # doctest: +SKIP
        >>>
        >>> # now demonstrate use of fromhdf5
        >>> table1 = etl.fromhdf5('example.h5', '/testgroup', 'testtable') # doctest: +SKIP
        >>> table1 # doctest: +SKIP
        +-----+-----------+
        | foo | bar       |
        +=====+===========+
        |   1 | b'asdfgh' |
        +-----+-----------+
        |   2 | b'qwerty' |
        +-----+-----------+
        |   3 | b'zxcvbn' |
        +-----+-----------+

        >>> # alternatively just specify path to table node
        ... table1 = etl.fromhdf5('example.h5', '/testgroup/testtable') # doctest: +SKIP
        >>> # ...or use an existing tables.File object
        ... h5file = tables.open_file('example.h5') # doctest: +SKIP
        >>> table1 = etl.fromhdf5(h5file, '/testgroup/testtable') # doctest: +SKIP
        >>> # ...or use an existing tables.Table object
        ... h5tbl = h5file.get_node('/testgroup/testtable') # doctest: +SKIP
        >>> table1 = etl.fromhdf5(h5tbl) # doctest: +SKIP
        >>> # use a condition to filter data
        ... table2 = etl.fromhdf5(h5tbl, condition='foo < 3') # doctest: +SKIP
        >>> table2 # doctest: +SKIP
        +-----+-----------+
        | foo | bar       |
        +=====+===========+
        |   1 | b'asdfgh' |
        +-----+-----------+
        |   2 | b'qwerty' |
        +-----+-----------+

        >>> h5file.close() # doctest: +SKIP

    """
    return HDF5View(source, where=where, name=name, condition=condition, condvars=condvars, start=start, stop=stop, step=step)