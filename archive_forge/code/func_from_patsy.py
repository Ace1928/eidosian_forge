import numpy as np
@classmethod
def from_patsy(cls, lc):
    """class method to create instance from patsy instance

        Parameters
        ----------
        lc : instance
            instance of patsy LinearConstraint, or other instances that have
            attributes ``lc.coefs, lc.constants, lc.variable_names``

        Returns
        -------
        instance of this class

        """
    return cls(lc.coefs, lc.constants, lc.variable_names)