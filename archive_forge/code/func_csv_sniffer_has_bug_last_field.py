import re
import datetime
import numpy as np
import csv
import ctypes
def csv_sniffer_has_bug_last_field():
    """
    Checks if the bug https://bugs.python.org/issue30157 is unpatched.
    """
    has_bug = getattr(csv_sniffer_has_bug_last_field, 'has_bug', None)
    if has_bug is None:
        dialect = csv.Sniffer().sniff("3, 'a'")
        csv_sniffer_has_bug_last_field.has_bug = dialect.quotechar != "'"
        has_bug = csv_sniffer_has_bug_last_field.has_bug
    return has_bug