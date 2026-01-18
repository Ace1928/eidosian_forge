from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
def test_complete_dictionary_subcmd(self):
    sot = complete.CompleteDictionary()
    sot.add_command('image delete'.split(), [mock.Mock(option_strings=['1'])])
    sot.add_command('image list'.split(), [mock.Mock(option_strings=['2'])])
    sot.add_command('image list better'.split(), [mock.Mock(option_strings=['3'])])
    self.assertEqual('image', sot.get_commands())
    result = sot.get_data()
    self.assertEqual('image', result[0][0])
    self.assertEqual('delete list list_better', result[0][1])
    self.assertEqual('image_delete', result[1][0])
    self.assertEqual('1', result[1][1])
    self.assertEqual('image_list', result[2][0])
    self.assertEqual('2 better', result[2][1])
    self.assertEqual('image_list_better', result[3][0])
    self.assertEqual('3', result[3][1])