import sys
import importlib
import cherrypy
from cherrypy.test import helper
def test03GetAndPost(self):
    self.setup_tutorial('tut03_get_and_post', 'WelcomePage')
    self.getPage('/greetUser?name=Bob')
    self.assertBody("Hey Bob, what's up?")
    self.getPage('/greetUser')
    self.assertBody('Please enter your name <a href="./">here</a>.')
    self.getPage('/greetUser?name=')
    self.assertBody('No, really, enter your name <a href="./">here</a>.')
    self.getPage('/greetUser', method='POST', body='name=Bob')
    self.assertBody("Hey Bob, what's up?")
    self.getPage('/greetUser', method='POST', body='name=')
    self.assertBody('No, really, enter your name <a href="./">here</a>.')