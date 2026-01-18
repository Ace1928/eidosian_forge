import yaml
import os
def is_exclude(exclude_list, test_suite):
    test_is_excluded = False
    for excl in exclude_list:
        match = 0
        if 'ansible' in excl:
            if excl.get('ansible') == test_suite.get('ansible'):
                match += 1
        if 'db_engine_name' in excl:
            if excl.get('db_engine_name') == test_suite.get('db_engine_name'):
                match += 1
        if 'db_engine_version' in excl:
            if excl.get('db_engine_version') == test_suite.get('db_engine_version'):
                match += 1
        if 'python' in excl:
            if excl.get('python') == test_suite.get('python'):
                match += 1
        if 'connector_name' in excl:
            if excl.get('connector_name') == test_suite.get('connector_name'):
                match += 1
        if 'connector_version' in excl:
            if excl.get('connector_version') == test_suite.get('connector_version'):
                match += 1
        if match > 1:
            test_is_excluded = True
            return test_is_excluded
    return test_is_excluded