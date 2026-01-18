from prov.model import *
def test_attribution_6(self):
    document = self.new_document()
    attr = document.attribution(EX_NS['e1'], EX_NS['ag1'], identifier=EX_NS['attr6'])
    self.add_labels(attr)
    self.do_tests(document)