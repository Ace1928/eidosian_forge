import io
import csv
import logging
from petl.util.base import Table, data
def teecsv_impl(table, source, **kwargs):
    return TeeCSVView(table, source=source, **kwargs)