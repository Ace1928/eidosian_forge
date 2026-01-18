from io import BytesIO
from ... import tests
from .. import pack
def assertRecordParsing(self, expected_record, data):
    """Assert that 'bytes' is parsed as a given bytes record.

        :param expected_record: A tuple of (names, bytes).
        """
    parser = self.make_parser_expecting_bytes_record()
    parser.accept_bytes(data)
    parsed_records = parser.read_pending_records()
    self.assertEqual([expected_record], parsed_records)