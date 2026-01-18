from prov.model import ProvDocument
def test_default_namespace_inheritance(self):
    prov_doc = ProvDocument()
    prov_doc.set_default_namespace('http://www.example.org/')
    bundle = prov_doc.bundle('bundle')
    e1 = bundle.entity('e1')
    self.assertIsNotNone(e1.identifier, "e1's identifier is None!")
    self.do_tests(prov_doc)