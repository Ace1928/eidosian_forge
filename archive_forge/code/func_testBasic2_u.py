from hashlib import md5
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.lib import auth_basic
from cherrypy.test import helper
def testBasic2_u(self):
    self.getPage('/basic2_u/')
    self.assertStatus(401)
    self.assertHeader('WWW-Authenticate', 'Basic realm="wonderland", charset="UTF-8"')
    self.getPage('/basic2_u/', [('Authorization', 'Basic eNGO0LfQtdGAOtGX0LbRgw==')])
    self.assertStatus(401)
    self.getPage('/basic2_u/', [('Authorization', 'Basic eNGO0LfQtdGAOtGX0LbQsA==')])
    self.assertStatus('200 OK')
    self.assertBody("Hello xюзер, you've been authorized.")