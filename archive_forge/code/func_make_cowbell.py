import os
import re
from paste.fileapp import FileApp
from paste.response import header_value, remove_header
def make_cowbell(global_conf, app):
    return MoreCowbell(app)