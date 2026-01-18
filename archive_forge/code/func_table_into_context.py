from parlai.core.teachers import DialogTeacher
from .build import build
import os
import json
def table_into_context(table):
    header = table['header']
    if len(header) == 0:
        return 'The table has no columns'
    elif len(header) == 1:
        return 'The table has column {}'.format(header[0])
    else:
        return 'The table has column names {} and {}.'.format(', '.join(header[:-1]), header[-1])