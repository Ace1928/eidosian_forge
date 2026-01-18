import numpy
def readReals(self, prec='f'):
    """Read in an array of real numbers.

        Parameters
        ----------
        prec : character, optional
            Specify the precision of the array using character codes from
            Python's struct module.  Possible values are 'd' and 'f'.

        """
    if prec not in self._real_precisions:
        raise ValueError('Not an appropriate precision')
    data_str = self.readRecord()
    return numpy.frombuffer(data_str, dtype=self.ENDIAN + prec)