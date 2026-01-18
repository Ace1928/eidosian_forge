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
def stylesheet_call(self, path):
    """Return code to reference or embed stylesheet file `path`"""
    base, ext = os.path.splitext(path)
    is_package = ext in ['.sty', '']
    if self.settings.embed_stylesheet:
        if is_package:
            path = base + '.sty'
        try:
            content = io.FileInput(source_path=path, encoding='utf-8').read()
            self.settings.record_dependencies.add(path)
        except IOError as err:
            msg = "Cannot embed stylesheet '%s':\n  %s." % (path, SafeString(err.strerror))
            self.document.reporter.error(msg)
            return '% ' + msg.replace('\n', '\n% ')
        if is_package:
            content = '\n'.join(['\\makeatletter', content, '\\makeatother'])
        return '%% embedded stylesheet: %s\n%s' % (path, content)
    if is_package:
        path = base
        cmd = '\\usepackage{%s}'
    else:
        cmd = '\\input{%s}'
    if self.settings.stylesheet_path:
        path = utils.relative_path(self.settings._destination, path)
    return cmd % path