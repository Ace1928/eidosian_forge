from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_unique_string(baseName):
    unique_string = baseName + datetime.datetime.now().strftime('-%d-%m-%Y') + '-' + str(uuid.uuid1().time)
    unique_string = unique_string[:63]
    return unique_string