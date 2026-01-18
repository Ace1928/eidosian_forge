import sys
import importlib
import cherrypy
from cherrypy.test import helper
def test10HTTPErrors(self):
    self.setup_tutorial('tut10_http_errors', 'HTTPErrorDemo')

    @cherrypy.expose
    def traceback_setting():
        return repr(cherrypy.request.show_tracebacks)
    cherrypy.tree.mount(traceback_setting, '/traceback_setting')
    self.getPage('/')
    self.assertInBody('<a href="toggleTracebacks">')
    self.assertInBody('<a href="/doesNotExist">')
    self.assertInBody('<a href="/error?code=403">')
    self.assertInBody('<a href="/error?code=500">')
    self.assertInBody('<a href="/messageArg">')
    self.getPage('/traceback_setting')
    setting = self.body
    self.getPage('/toggleTracebacks')
    self.assertStatus((302, 303))
    self.getPage('/traceback_setting')
    self.assertBody(str(not eval(setting)))
    self.getPage('/error?code=500')
    self.assertStatus(500)
    self.assertInBody('The server encountered an unexpected condition which prevented it from fulfilling the request.')
    self.getPage('/error?code=403')
    self.assertStatus(403)
    self.assertInBody("<h2>You can't do that!</h2>")
    self.getPage('/messageArg')
    self.assertStatus(500)
    self.assertInBody("If you construct an HTTPError with a 'message'")