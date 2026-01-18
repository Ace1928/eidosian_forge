import os
import os.path
import cherrypy
@cherrypy.expose
def messageArg(self):
    message = "If you construct an HTTPError with a 'message' argument, it wil be placed on the error page (underneath the status line by default)."
    raise cherrypy.HTTPError(500, message=message)