import logging
import os
import sys
import threading
import time
import cherrypy
from cherrypy._json import json
def pause_resume(ns):

    def _pause_resume(enabled):
        pause_disabled = ''
        resume_disabled = ''
        if enabled:
            resume_disabled = 'disabled="disabled" '
        else:
            pause_disabled = 'disabled="disabled" '
        return '\n            <form action="pause" method="POST" style="display:inline">\n            <input type="hidden" name="namespace" value="%s" />\n            <input type="submit" value="Pause" %s/>\n            </form>\n            <form action="resume" method="POST" style="display:inline">\n            <input type="hidden" name="namespace" value="%s" />\n            <input type="submit" value="Resume" %s/>\n            </form>\n            ' % (ns, pause_disabled, ns, resume_disabled)
    return _pause_resume