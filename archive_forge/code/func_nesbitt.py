import cherrypy
from cherrypy import expose, tools
@expose('call_alias')
def nesbitt(self):
    return 'Mr Nesbitt'