import os
from collections import namedtuple
import re
import sqlite3
import typing
import warnings
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects.packages_utils import (get_packagepath,
from collections import OrderedDict
def populate_metaRd_db(package_name: str, dbcon, package_path: typing.Optional[str]=None) -> None:
    """ Populate a database with the meta-information
    associated with an R package: version, description, title, and
    aliases (those are what the R help system is organised around).

    - package_name: a string
    - dbcon: a database connection
    - package_path: path the R package installation (default: None)
    """
    if package_path is None:
        package_path = get_packagepath(package_name)
    rpath = StrSexpVector((os.path.join(package_path, __package_meta),))
    rds = readRDS(rpath)
    desc = rds[rds.do_slot('names').index('DESCRIPTION')]
    db_res = dbcon.execute('insert into package values (?,?,?,?)', (desc[desc.do_slot('names').index('Package')], desc[desc.do_slot('names').index('Title')], desc[desc.do_slot('names').index('Version')], desc[desc.do_slot('names').index('Description')]))
    package_rowid = db_res.lastrowid
    rpath = StrSexpVector((os.path.join(package_path, __rd_meta),))
    rds = readRDS(rpath)
    FILE_I = rds.do_slot('names').index('File')
    NAME_I = rds.do_slot('names').index('Name')
    TYPE_I = rds.do_slot('names').index('Type')
    TITLE_I = rds.do_slot('names').index('Title')
    ENCODING_I = rds.do_slot('names').index('Encoding')
    ALIAS_I = rds.do_slot('names').index('Aliases')
    for row_i in range(len(rds[0])):
        db_res = dbcon.execute('insert into rd_meta values (?,?,?,?,?,?,?)', (row_i, rds[FILE_I][row_i], rds[NAME_I][row_i], rds[TYPE_I][row_i], rds[TITLE_I][row_i], rds[ENCODING_I][row_i], package_rowid))
        rd_rowid = db_res.lastrowid
        for alias in rds[ALIAS_I][row_i]:
            dbcon.execute('insert into rd_alias_meta values (?,?)', (rd_rowid, alias))