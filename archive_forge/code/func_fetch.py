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
def fetch(self, alias: str) -> Page:
    """ Fetch the documentation page associated with a given alias.

        For S4 classes, the class name is *often* suffixed with '-class'.
        For example, the alias to the documentation for the class
        AnnotatedDataFrame in the package Biobase is
        'AnnotatedDataFrame-class'.
        """
    c = self._dbcon.execute('SELECT rd_meta_rowid, alias FROM rd_alias_meta WHERE alias=?', (alias,))
    res_alias = c.fetchall()
    if len(res_alias) == 0:
        raise HelpNotFoundError('No help could be fetched', topic=alias, package=self.__package_name)
    c = self._dbcon.execute('SELECT file, name, type FROM rd_meta WHERE rowid=?', (res_alias[0][0],))
    res_all = c.fetchall()
    rkey = StrSexpVector((res_all[0][0][:-3],))
    _type = res_all[0][2]
    rpath = StrSexpVector((os.path.join(self.package_path, 'help', f'{self.__package_name}.rdb'),))
    rdx_variables = self._rdx[self._rdx.do_slot('names').index('variables')]
    _eval = rinterface.baseenv['eval']
    devnull_func = rinterface.parse('function(x) {}')
    devnull_func = _eval(devnull_func)
    res = _lazyload_dbfetch(rdx_variables[rdx_variables.do_slot('names').index(rkey[0])], rpath, self._rdx[self._rdx.do_slot('names').index('compressed')], devnull_func)
    p_res = Page(res, _type=_type)
    return p_res