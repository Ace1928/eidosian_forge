import warnings
def assert_all_input_consumed(self):
    if self.responses:
        raise AssertionError('expected all input in %r to be consumed' % (self,))