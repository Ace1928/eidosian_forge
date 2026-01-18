from prov.model import *
def test_attribution_8(self):
    document = self.new_document()
    attr = document.attribution(EX_NS['e1'], EX_NS['ag1'], identifier=EX_NS['attr8'])
    self.add_labels(attr)
    self.add_types(attr)
    self.add_further_attributes(attr)
    self.do_tests(document)