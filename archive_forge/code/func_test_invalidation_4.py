from prov.model import *
def test_invalidation_4(self):
    document = self.new_document()
    inv = document.invalidation(EX_NS['e1'], identifier=EX_NS['inv4'], activity=EX_NS['a1'], time=datetime.datetime.now())
    inv.add_attributes([(PROV_ROLE, 'someRole')])
    self.do_tests(document)