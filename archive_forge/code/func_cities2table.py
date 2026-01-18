import os
import re
import shelve
import sys
import nltk.data
def cities2table(filename, rel_name, dbname, verbose=False, setup=False):
    """
    Convert a file of Prolog clauses into a database table.

    This is not generic, since it doesn't allow arbitrary
    schemas to be set as a parameter.

    Intended usage::

        cities2table('cities.pl', 'city', 'city.db', verbose=True, setup=True)

    :param filename: filename containing the relations
    :type filename: str
    :param rel_name: name of the relation
    :type rel_name: str
    :param dbname: filename of persistent store
    :type schema: str
    """
    import sqlite3
    records = _str2records(filename, rel_name)
    connection = sqlite3.connect(dbname)
    cur = connection.cursor()
    if setup:
        cur.execute('CREATE TABLE city_table\n        (City text, Country text, Population int)')
    table_name = 'city_table'
    for t in records:
        cur.execute('insert into %s values (?,?,?)' % table_name, t)
        if verbose:
            print('inserting values into %s: ' % table_name, t)
    connection.commit()
    if verbose:
        print('Committing update to %s' % dbname)
    cur.close()