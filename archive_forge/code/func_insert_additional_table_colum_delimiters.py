import sys
import os
import time
import re
import string
import urllib.request, urllib.parse, urllib.error
from docutils import frontend, nodes, languages, writers, utils, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import pick_math_environment, unichar2tex
def insert_additional_table_colum_delimiters(self):
    while self.active_table.get_rowspan(self.active_table.get_entry_number()):
        self.out.append(' & ')
        self.active_table.visit_entry()