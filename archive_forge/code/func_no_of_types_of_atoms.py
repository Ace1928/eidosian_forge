import numpy as np
@property
def no_of_types_of_atoms(self):
    """

        Dynamically returns the number of different atoms in the system
        
        """
    return len(self.types_of_atoms)