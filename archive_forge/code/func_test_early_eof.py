from io import BytesIO
from ... import tests
from .. import pack
def test_early_eof(self):
    """Tests for premature EOF occuring during parsing Bytes records with
        BytesRecordReader.

        A incomplete container might be interrupted at any point.  The
        BytesRecordReader needs to cope with the input stream running out no
        matter where it is in the parsing process.

        In all cases, UnexpectedEndOfContainerError should be raised.
        """
    complete_record = b'6\nname\n\nabcdef'
    for count in range(0, len(complete_record)):
        incomplete_record = complete_record[:count]
        reader = self.get_reader_for(incomplete_record)
        try:
            names, read_bytes = reader.read()
            read_bytes(None)
        except pack.UnexpectedEndOfContainerError:
            pass
        else:
            self.fail('UnexpectedEndOfContainerError not raised when parsing %r' % (incomplete_record,))