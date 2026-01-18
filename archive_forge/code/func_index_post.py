from pecan import expose, redirect
from webob.exc import status_map
@index.when(method='POST')
def index_post(self, q):
    redirect('https://pecan.readthedocs.io/en/latest/search.html?q=%s' % q)