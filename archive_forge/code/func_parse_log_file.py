import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def parse_log_file(self):
    logfilename = os.path.splitext(self.tempfilename)[0] + '.log'
    tmpdir = os.getcwd()
    os.chdir(os.path.split(logfilename)[0])
    if self.options.get('usepdflatex'):
        command = 'pdflatex -interaction=nonstopmode %s' % self.tempfilename
    else:
        command = 'latex -interaction=nonstopmode %s' % self.tempfilename
    log.debug('Running command: %s' % command)
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, close_fds=sys.platform != 'win32')
    stdout, stderr = (p.stdout, p.stderr)
    try:
        data = stdout.read()
        log.debug('stdout from latex\n %s', data)
    finally:
        stdout.close()
    try:
        error_data = stderr.read()
        if error_data:
            log.debug('latex STDERR %s', error_data)
    finally:
        stderr.close()
    p.kill()
    p.wait()
    with open(logfilename, 'r') as f:
        logdata = f.read()
    log.debug('Logfile from LaTeX run: \n' + logdata)
    os.chdir(tmpdir)
    texdimdata = self.dimext_re.findall(logdata)
    log.debug('Texdimdata: ' + str(texdimdata))
    if len(texdimdata) == 0:
        log.error('No dimension data could be extracted from dot2tex.tex.')
        self.texdims = None
        return
    c = 1.0 / 4736286
    self.texdims = {}
    self.texdimlist = [(float(i[1]) * c, float(i[2]) * c, float(i[3]) * c) for i in texdimdata]
    self.texdims = dict(zip(self.snippets_id, self.texdimlist))