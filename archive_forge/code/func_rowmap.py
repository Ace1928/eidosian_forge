from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.compat import next, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError
from petl.util.base import Table, expr, rowgroupby, Record
from petl.transform.sorts import sort
def rowmap(table, rowmapper, header, failonerror=None):
    """
    Transform rows via an arbitrary function. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'sex', 'age', 'height', 'weight'],
        ...           [1, 'male', 16, 1.45, 62.0],
        ...           [2, 'female', 19, 1.34, 55.4],
        ...           [3, 'female', 17, 1.78, 74.4],
        ...           [4, 'male', 21, 1.33, 45.2],
        ...           [5, '-', 25, 1.65, 51.9]]
        >>> def rowmapper(row):
        ...     transmf = {'male': 'M', 'female': 'F'}
        ...     return [row[0],
        ...             transmf[row['sex']] if row['sex'] in transmf else None,
        ...             row.age * 12,
        ...             row.height / row.weight ** 2]
        ...
        >>> table2 = etl.rowmap(table1, rowmapper,
        ...                     header=['subject_id', 'gender', 'age_months',
        ...                             'bmi'])
        >>> table2
        +------------+--------+------------+-----------------------+
        | subject_id | gender | age_months | bmi                   |
        +============+========+============+=======================+
        |          1 | 'M'    |        192 | 0.0003772112382934443 |
        +------------+--------+------------+-----------------------+
        |          2 | 'F'    |        228 | 0.0004366015456998006 |
        +------------+--------+------------+-----------------------+
        |          3 | 'F'    |        204 | 0.0003215689675106949 |
        +------------+--------+------------+-----------------------+
        |          4 | 'M'    |        252 | 0.0006509906805544679 |
        +------------+--------+------------+-----------------------+
        |          5 | None   |        300 | 0.0006125608384287258 |
        +------------+--------+------------+-----------------------+

    The `rowmapper` function should accept a single row and return a single
    row (list or tuple).

    The `failonerror` keyword argument is documented under
    :func:`petl.config.failonerror`
    """
    return RowMapView(table, rowmapper, header, failonerror=failonerror)