from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
def add_spec(sub_it):
    it = iter(sub_it)
    try:
        yield ('%s%s%s' % (self.spec, self.sep, next(it)))
    except StopIteration:
        pass
    for line in it:
        yield line