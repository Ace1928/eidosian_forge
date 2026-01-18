import functools
import itertools
import logging
import os
import re
import time
import debtcollector.removals
import debtcollector.renames
import sqlalchemy
from sqlalchemy import event
from sqlalchemy import exc
from sqlalchemy import pool
from sqlalchemy import select
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
@sqlalchemy.event.listens_for(engine, 'before_cursor_execute', retval=True)
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    stack = traceback.extract_stack()
    our_line = None
    for idx, (filename, line, method, function) in enumerate(stack):
        for tgt in skip_paths:
            if filename.startswith(tgt):
                break
        else:
            for tgt in target_paths:
                if filename.startswith(tgt):
                    our_line = idx
                    break
        if our_line:
            break
    if our_line:
        trace = '; '.join(('File: %s (%s) %s' % (line[0], line[1], line[2]) for line in stack[our_line - 3:our_line]))
        statement = '%s  -- %s' % (statement, trace)
    return (statement, parameters)