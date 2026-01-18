import os
import cherrypy
from cherrypy.test import helper
def testVirtualHost(self):
    self.getPage('/', [('Host', 'www.mydom1.com')])
    self.assertBody('Hello, world')
    self.getPage('/mydom2/', [('Host', 'www.mydom1.com')])
    self.assertBody('Welcome to Domain 2')
    self.getPage('/', [('Host', 'www.mydom2.com')])
    self.assertBody('Welcome to Domain 2')
    self.getPage('/', [('Host', 'www.mydom3.com')])
    self.assertBody('Welcome to Domain 3')
    self.getPage('/', [('Host', 'www.mydom4.com')])
    self.assertBody('Under construction')
    self.getPage('/method?value=root')
    self.assertBody('You sent root')
    self.getPage('/vmethod?value=dom2+GET', [('Host', 'www.mydom2.com')])
    self.assertBody('You sent dom2 GET')
    self.getPage('/vmethod', [('Host', 'www.mydom3.com')], method='POST', body='value=dom3+POST')
    self.assertBody('You sent dom3 POST')
    self.getPage('/vmethod/pos', [('Host', 'www.mydom3.com')])
    self.assertBody('You sent pos')
    self.getPage('/url', [('Host', 'www.mydom2.com')])
    self.assertBody('%s://www.mydom2.com/nextpage' % self.scheme)