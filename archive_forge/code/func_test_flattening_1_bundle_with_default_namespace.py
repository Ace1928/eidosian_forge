from prov.model import ProvDocument
def test_flattening_1_bundle_with_default_namespace(self):
    prov_doc = document_with_n_bundles_having_default_namespace(1)
    flattened = prov_doc.flattened()
    self.do_tests(flattened)