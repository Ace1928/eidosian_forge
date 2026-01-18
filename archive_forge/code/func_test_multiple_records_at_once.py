from io import BytesIO
from ... import tests
from .. import pack
def test_multiple_records_at_once(self):
    """If multiple records worth of data are fed to the parser in one
        string, the parser will correctly parse all the records.

        (A naive implementation might stop after parsing the first record.)
        """
    parser = self.make_parser_expecting_record_type()
    parser.accept_bytes(b'B5\nname1\n\nbody1B5\nname2\n\nbody2')
    self.assertEqual([([(b'name1',)], b'body1'), ([(b'name2',)], b'body2')], parser.read_pending_records())