from ..views import FileOutsideView, NoSuchView, ViewsNotSupported
from . import TestCase
def test_file_outside_view(self):
    err = FileOutsideView('baz', ['foo', 'bar'])
    self.assertEqual('Specified file "baz" is outside the current view: foo, bar', str(err))