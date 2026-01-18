import re
import sys
import cgi
import os
import os.path
import urllib.parse
import cherrypy
def annotated_file(self, filename, statements, excluded, missing):
    with open(filename, 'r') as source:
        lines = source.readlines()
    buffer = []
    for lineno, line in enumerate(lines):
        lineno += 1
        line = line.strip('\n\r')
        empty_the_buffer = True
        if lineno in excluded:
            template = TEMPLATE_LOC_EXCLUDED
        elif lineno in missing:
            template = TEMPLATE_LOC_NOT_COVERED
        elif lineno in statements:
            template = TEMPLATE_LOC_COVERED
        else:
            empty_the_buffer = False
            buffer.append((lineno, line))
        if empty_the_buffer:
            for lno, pastline in buffer:
                yield (template % (lno, cgi.escape(pastline)))
            buffer = []
            yield (template % (lineno, cgi.escape(line)))