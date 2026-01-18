from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
def get_DT_tables():
    """
    Returns two barebones databases for looking up DT codes by name. 
    """

    class DTCodeTable(object):
        """
        A barebones database for looking up a DT code by knot/link name.
        """

        def __init__(self, name='', table='', db_path=database_path, **filter_args):
            self._table = table
            self._select = 'select DT from {}'.format(table)
            self.name = name
            self._connection = connect_to_db(db_path)
            self._cursor = self._connection.cursor()

        def __repr__(self):
            return self.name

        def __getitem__(self, link_name):
            select_query = self._select + ' where name="{}"'.format(link_name)
            return self._cursor.execute(select_query).fetchall()[0][0]

        def __len__(self):
            length_query = 'select count(*) from ' + self._table
            return self._cursor.execute(length_query).fetchone()[0]
    RolfsenDTcodes = DTCodeTable(name='RolfsenDTcodes', table='link_exteriors', db_path=database_path)
    HTLinkDTcodes = DTCodeTable(name='HTLinkDTcodes', table='HT_links', db_path=alt_database_path)
    return [RolfsenDTcodes, HTLinkDTcodes]