import warnings
import re
import string
import itertools
from Bio.Seq import Seq, MutableSeq
from Bio.Restriction.Restriction_Dictionary import rest_dict as enzymedict
from Bio.Restriction.Restriction_Dictionary import typedict
from Bio.Restriction.Restriction_Dictionary import suppliers as suppliers_dict
from Bio.Restriction.PrintFormat import PrintFormat
from Bio import BiopythonWarning
def add_supplier(self, letter):
    """Add all enzymes from a given supplier to batch.

        letter represents the suppliers as defined in the dictionary
        RestrictionDictionary.suppliers
        Returns None.
        Raise a KeyError if letter is not a supplier code.
        """
    supplier = suppliers_dict[letter]
    self.suppliers.append(letter)
    for x in supplier[1]:
        self.add_nocheck(eval(x))