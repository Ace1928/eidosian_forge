from prov.model import ProvDocument
def test_namespace_inheritance(self):
    prov_doc = ProvDocument()
    prov_doc.add_namespace('ex', 'http://www.example.org/')
    bundle = prov_doc.bundle('ex:bundle')
    e1 = bundle.entity('ex:e1')
    self.assertIsNotNone(e1.identifier, "e1's identifier is None!")
    self.do_tests(prov_doc)