from prov.model import *
def test_end_8(self):
    document = self.new_document()
    end = document.end(EX_NS['a1'], identifier=EX_NS['end8'], ender=EX_NS['a2'], time=datetime.datetime.now())
    end.add_attributes([(PROV_ROLE, 'egg-cup'), (PROV_ROLE, 'boiling-water')])
    self.add_types(end)
    self.add_locations(end)
    self.add_labels(end)
    self.add_further_attributes(end)
    self.do_tests(document)