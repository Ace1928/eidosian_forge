from prov.model import *
def test_invalidation_5(self):
    document = self.new_document()
    inv = document.invalidation(EX_NS['e1'], identifier=EX_NS['inv5'], activity=EX_NS['a1'], time=datetime.datetime.now())
    inv.add_attributes([(PROV_ROLE, 'someRole')])
    self.add_types(inv)
    self.add_locations(inv)
    self.add_labels(inv)
    self.add_further_attributes(inv)
    self.do_tests(document)