from prov.model import *
def test_attribution_3(self):
    document = self.new_document()
    document.attribution(EX_NS['e1'], EX_NS['ag1'], identifier=EX_NS['attr3'])
    self.do_tests(document)