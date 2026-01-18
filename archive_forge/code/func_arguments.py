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
def arguments(self) -> typing.List[Item]:
    """ Get the arguments and descriptions as a list of Item objects. """
    section_doc = self._sections.get('\\arguments')
    res: typing.List[Item] = list()
    if section_doc is None:
        return res
    else:
        arg_name = None
        arg_desc = None
        section_rows = _Rd2txt(section_doc)
        if len(section_rows) < 3:
            return res
        for row in section_rows[2:]:
            if arg_name is None:
                m = p_newarg.match(row)
                if m:
                    arg_name = m.groups()[0]
                    arg_desc = [m.groups()[1]]
            elif p_desc.match(row):
                arg_desc.append(row.strip())
            else:
                res.append(Item(arg_name, arg_desc))
                arg_name = None
                arg_desc = None
        if arg_name is not None:
            res.append(Item(arg_name, arg_desc))
    return res