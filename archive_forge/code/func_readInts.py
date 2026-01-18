import numpy
def readInts(self, prec='i'):
    """Read an array of integers.

        Parameters
        ----------
        prec : character, optional
            Specify the precision of the data to be read using
            character codes from Python's struct module.  Possible
            values are 'h', 'i', 'l' and 'q'

        """
    if prec not in self._int_precisions:
        raise ValueError('Not an appropriate precision')
    data_str = self.readRecord()
    return numpy.frombuffer(data_str, dtype=self.ENDIAN + prec)