from __future__ import absolute_import, division, print_function
import datetime
import uuid
def remove_null_args(**kwargs):
    tosearch = kwargs.copy()
    for key, value in tosearch.items():
        if type(value) is not bool and type(value) is not list:
            if is_null_or_empty(value):
                kwargs.pop(key)
    return kwargs