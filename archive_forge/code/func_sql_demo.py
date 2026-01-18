import os
import re
import shelve
import sys
import nltk.data
def sql_demo():
    """
    Print out every row from the 'city.db' database.
    """
    print()
    print("Using SQL to extract rows from 'city.db' RDB.")
    for row in sql_query('corpora/city_database/city.db', 'SELECT * FROM city_table'):
        print(row)