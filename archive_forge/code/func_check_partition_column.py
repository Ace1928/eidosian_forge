import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from modin.core.storage_formats.pandas.parsers import _split_result_for_readers
def check_partition_column(partition_column, cols):
    """
    Check `partition_column` existence and it's type.

    Parameters
    ----------
    partition_column : str
        Column name used for data partitioning between the workers.
    cols : dict
        Dictionary with columns names and python types.
    """
    for k, v in cols.items():
        if k == partition_column:
            if v == 'int':
                return
            raise InvalidPartitionColumn(f'partition_column must be int, and not {v}')
    raise InvalidPartitionColumn(f'partition_column {partition_column} not found in the query')