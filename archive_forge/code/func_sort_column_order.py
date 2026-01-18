from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def sort_column_order(statement):
    """Sort column order in grants like SELECT (colA, colB, ...).

    MySQL changes columns order like below:
    ---------------------------------------
    mysql> GRANT SELECT (testColA, testColB), INSERT ON `testDb`.`testTable` TO 'testUser'@'localhost';
    Query OK, 0 rows affected (0.04 sec)

    mysql> flush privileges;
    Query OK, 0 rows affected (0.00 sec)

    mysql> SHOW GRANTS FOR testUser@localhost;
    +---------------------------------------------------------------------------------------------+
    | Grants for testUser@localhost                                                               |
    +---------------------------------------------------------------------------------------------+
    | GRANT USAGE ON *.* TO 'testUser'@'localhost'                                                |
    | GRANT SELECT (testColB, testColA), INSERT ON `testDb`.`testTable` TO 'testUser'@'localhost' |
    +---------------------------------------------------------------------------------------------+

    We should sort columns in our statement, otherwise the module always will return
    that the state has changed.
    """
    tmp = statement.split('(')
    priv_name = tmp[0]
    columns = tmp[1].rstrip(')')
    columns = columns.split(',')
    for i, col in enumerate(columns):
        col = col.strip()
        columns[i] = col.strip('`')
    columns.sort()
    return '%s(%s)' % (priv_name, ', '.join(columns))