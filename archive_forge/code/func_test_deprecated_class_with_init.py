from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch('oslo_log.versionutils.report_deprecated_feature')
def test_deprecated_class_with_init(self, mock_reporter):
    mock_arguments = mock.MagicMock()
    args = (1, 5, 7)
    kwargs = {'first': 10, 'second': 20}

    @versionutils.deprecated(as_of=versionutils.deprecated.JUNO, remove_in=+1)
    class OutdatedClass(object):

        def __init__(self, *args, **kwargs):
            """It is __init__ method."""
            mock_arguments.args = args
            mock_arguments.kwargs = kwargs
            super(OutdatedClass, self).__init__()
    obj = OutdatedClass(*args, **kwargs)
    self.assertIsInstance(obj, OutdatedClass)
    self.assertEqual('__init__', obj.__init__.__name__)
    self.assertEqual('It is __init__ method.', obj.__init__.__doc__)
    self.assertEqual(args, mock_arguments.args)
    self.assertEqual(kwargs, mock_arguments.kwargs)
    self.assert_deprecated(mock_reporter, what='OutdatedClass()', as_of='Juno', remove_in='Kilo')