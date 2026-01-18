import cherrypy
from cherrypy.test import helper
def testSessionAuthenticate(self):
    self.getPage('/')
    self.assertInBody('<form method="post" action="do_login">')
    login_body = 'username=test&password=password&from_page=/'
    self.getPage('/do_login', method='POST', body=login_body)
    self.assertStatus((302, 303))
    self.getPage('/', self.cookies)
    self.assertBody('Hi test, you are logged in')
    self.getPage('/do_logout', self.cookies, method='POST')
    self.assertStatus((302, 303))
    self.getPage('/', self.cookies)
    self.assertInBody('<form method="post" action="do_login">')