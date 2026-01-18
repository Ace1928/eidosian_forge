from prov.model import *
def test_attribution_2(self):
    document = self.new_document()
    document.attribution(None, EX_NS['ag1'], identifier=EX_NS['attr2'])
    self.do_tests(document)