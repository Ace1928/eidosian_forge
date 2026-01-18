@data_function.setter
def data_function(self, value):
    if value not in self.ValidDataFunctions:
        valid = '|'.join(self.ValidDataFunctions)
        raise ValueError('data_function must be one of: %s' % valid)
    self._data_function = value