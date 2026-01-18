import unittest
from jsbeautifier.unpackers.urlencode import detect, unpack
def test_detect(self):
    """Test detect() function."""

    def encoded(source):
        return self.assertTrue(detect(source))

    def unencoded(source):
        return self.assertFalse(detect(source))
    unencoded('')
    unencoded('var a = b')
    encoded('var%20a+=+b')
    encoded('var%20a=b')
    encoded('var%20%21%22')