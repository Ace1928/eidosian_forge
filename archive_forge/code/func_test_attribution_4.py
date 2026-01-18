from prov.model import *
def test_attribution_4(self):
    document = self.new_document()
    document.attribution(EX_NS['e1'], EX_NS['ag1'], identifier=EX_NS['attr4'])
    self.do_tests(document)