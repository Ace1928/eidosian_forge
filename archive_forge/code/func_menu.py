import io
import os
import os.path
import sys
import warnings
import cherrypy
@cherrypy.expose
def menu(self):
    yield '<h2>Profiling runs</h2>'
    yield '<p>Click on one of the runs below to see profiling data.</p>'
    runs = self.statfiles()
    runs.sort()
    for i in runs:
        yield ("<a href='report?filename=%s' target='main'>%s</a><br />" % (i, i))