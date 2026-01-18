import textwrap
@property
def offender(self):
    index = [x is not None for x in self.cause['got']].index(True)
    return (index, self._master_name(index))