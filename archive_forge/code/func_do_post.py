from json import dumps
from webtest import TestApp
from pecan import Pecan, expose, abort
from pecan.tests import PecanTestCase
@index.when(method='POST', template='json')
def do_post(self):
    return dict(result='POST')