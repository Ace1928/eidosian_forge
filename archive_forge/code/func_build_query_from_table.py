import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from modin.core.storage_formats.pandas.parsers import _split_result_for_readers
def build_query_from_table(name):
    """
    Create a query from the given table name.

    Parameters
    ----------
    name : str
        Table name.

    Returns
    -------
    str
        Query string.
    """
    return 'SELECT * FROM {0}'.format(name)