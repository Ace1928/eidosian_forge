import os
import re
import shelve
import sys
import nltk.data
def sql_query(dbname, query):
    """
    Execute an SQL query over a database.
    :param dbname: filename of persistent store
    :type schema: str
    :param query: SQL query
    :type rel_name: str
    """
    import sqlite3
    try:
        path = nltk.data.find(dbname)
        connection = sqlite3.connect(str(path))
        cur = connection.cursor()
        return cur.execute(query)
    except (ValueError, sqlite3.OperationalError):
        import warnings
        warnings.warn('Make sure the database file %s is installed and uncompressed.' % dbname)
        raise