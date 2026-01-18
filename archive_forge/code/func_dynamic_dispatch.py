import cherrypy
from cherrypy.test import helper
def dynamic_dispatch(self, vpath):
    try:
        id = int(vpath[0])
    except (ValueError, IndexError):
        return None
    return UserInstanceNode(id)