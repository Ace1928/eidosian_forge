import io
import logging
import unittest
from prov.model import ProvDocument
def assertRoundTripEquivalence(self, prov_doc, msg=None):
    if self.FORMAT is None:
        return
    with io.BytesIO() as stream:
        prov_doc.serialize(destination=stream, format=self.FORMAT, indent=4)
        stream.seek(0, 0)
        prov_doc_new = ProvDocument.deserialize(source=stream, format=self.FORMAT)
        stream.seek(0, 0)
        msg_extra = "'%s' serialization content:\n%s" % (self.FORMAT, stream.read().decode('utf-8'))
        msg = '\n'.join((msg, msg_extra)) if msg else msg_extra
        self.assertEqual(prov_doc, prov_doc_new, msg)