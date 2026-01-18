from prov.model import ProvDocument
def test_flattening_3_bundles_with_default_namespace(self):
    prov_doc = document_with_n_bundles_having_default_namespace(3)
    flattened = prov_doc.flattened()
    self.do_tests(flattened)